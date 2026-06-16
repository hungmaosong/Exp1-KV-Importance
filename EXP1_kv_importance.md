# Experiment 1：KV Importance / Hot-Cold Stability Analysis

## 0. Working Directory and Scope

本實驗會在 server 上的以下資料夾內進行：

```text
exp1_kv_importance/
```

請所有程式、資料、輸出結果都限制在此資料夾內產生，不要修改此資料夾以外的任何檔案。

## 1. 實驗目的

本實驗的目的是觀察 LLM 在 decode 過程中，不同 token 位置的 KV cache 重要程度是否固定，或是否會隨著 decode step 與 request type 改變。

具體來說，本實驗想回答以下問題：

1. 模型在生成新 token 時，attention 是否主要集中在 Sink KV 和 Recent KV？
2. Middle / Older tokens 對應的 Cold KV 是否在某些 request 中也會變重要？
3. Top-k important KV chunks 是否會隨著 decode step 改變？
4. 固定式 Hot / Cold KV 劃分是否足夠？
5. 是否需要 dynamic hotness tracking 或 request-aware Cold-KV retrieval?

## 2. 實驗核心想法

本實驗使用 full-KV inference 作為觀察基準。在不移除任何 KV cache 的情況下，記錄模型在每個 decode step 的 attention distribution。

Attention weight 可以作為 KV importance 的近似訊號，因為它代表模型在產生目前 token 時，對過去哪些 token 的 Key / Value 比較關注。

本實驗會先取得 token-level attention importance，也就是每個 decode step 中，每個過去 token position 收到的 attention weight。接著，再將 token-level importance 聚合成 chunk-level importance。

之所以需要 chunk-level importance，是因為未來若要做 KV cache offloading 或 Cold-KV retrieval，系統通常不會以單一 token 為搬移與管理單位，而會以一段連續 token range 作為 chunk 來管理。例如每 128、256 或 512 tokens 形成一個 KV chunk。

因此，本實驗的分析流程是：

```text
Full-KV inference
→ collect attention weights at each decode step
→ compute token-level importance
→ aggregate token-level importance into chunk-level importance
→ analyze whether hot chunks are stable or dynamic
```

## 3. 名詞與計算定義

### 3.1 Token-level importance

在 decode step `t`，模型會對過去所有 token positions 產生 attention weights。

本實驗將某個 token position `i` 在 decode step `t` 收到的 attention weight 視為該 token 的 KV importance：

```text
token_importance[t, i] = attention weight assigned to token position i at decode step t
```

如果模型有多個 layers 和 attention heads，請先對 layers 和 heads 做 mean，得到每個 token position 的平均 attention importance；若使用 sum，則必須在計算 Sink / Recent / Cold mass 前，對所有 valid token positions 重新 normalize，使總和為 1，得到每個 token position 的整體 importance。

---

### 3.2 Chunk-level importance

將 token positions 依照固定大小切成 chunks。

例如 `chunk_size = 256` 時：

```text
Chunk 0: token 0–255
Chunk 1: token 256–511
Chunk 2: token 512–767
Chunk 3: token 768–1023
...
```

在 decode step `t`，chunk `c` 的 importance 定義為該 chunk 內所有 token importance 的總和：

```text
chunk_importance[t, c]
= sum(token_importance[t, i]) for all token i inside chunk c
```

也就是說，token-level attention 是原始觀察訊號，而 chunk-level importance 是後續分析 Hot / Cold KV 穩定性的主要單位。

---

### 3.3 Sink / Recent / Cold 區域定義

本實驗將 token positions 分成三個區域：

```text
Sink KV   = 開頭 sink_size 個 tokens
Recent KV = decode step 當下最近 recent_window 個 tokens
Cold KV   = Sink KV 和 Recent KV 中間的 tokens
```

例如：

```text
sink_size = 8
recent_window = 1024
```

則：

```text
token 0–7             → Sink KV
最後 1024 個 tokens    → Recent KV
中間其他 tokens        → Cold KV
```

每個 decode step 都需要計算 attention mass 分布在 Sink KV、Recent KV 和 Cold KV 的比例。

## 4. 輸入資料格式與 Request 類型

本實驗使用 JSONL 格式作為輸入資料。請建立以下檔案：

```text
data/samples.jsonl
```

每一行代表一筆 sample，格式如下：

```json
{
  "id": "sample_001",
  "context": "context text ...",
  "query": "question or instruction ...",
  "type": "local"
}
```

欄位說明：

| 欄位      | 說明                                      |
| --------- | ------------------------------------     |
| `id`      | sample 的唯一識別名稱                     |
| `context` | 上下文內容                                |
| `query`   | 使用者問題或指令                           |
| `type`    | request 類型，可為 `local` 或 `long_range` |

---

### 4.1 Local request

`local` request 代表答案主要依賴最近上下文，也就是 Recent KV 應該就能提供足夠資訊。

範例：

```json
{
  "id": "local_001",
  "context": "前面有很多背景內容......最後一段提到：本系統主要使用 Hot KV 來降低 decode latency。",
  "query": "請總結最後一段的重點。",
  "type": "local"
}
```

預期觀察：

```text
attention mass 可能主要集中在 Recent KV。
```

---

### 4.2 Long-range request

`long_range` request 代表答案需要 early 或 middle context，也就是 Cold KV 可能會變重要。

範例：

```json
{
  "id": "long_001",
  "context": "文件開頭提到：系統代號是 CXL-KV-2026。中間有大量無關內容......最後是一般結論。",
  "query": "文件開頭提到的系統代號是什麼？",
  "type": "long_range"
}
```

預期觀察：

```text
某些 early 或 middle Cold KV chunks 可能在 decode 過程中獲得較高 attention。
```

---

### 4.3 Debug sample

請在 `data/samples.jsonl` 中先建立少量可測試資料，例如：

```text
2 筆 local request
2 筆 long_range request
```

初期資料不需要很大，先確保程式可以正確收集 attention、輸出 CSV 和產生圖表。

## 5. 模型與執行設定

本實驗使用 Hugging Face Transformers 和 PyTorch 實作。

由於本實驗的主要目標是觀察 KV importance，而不是一開始就追求超長 context，因此可以先用小模型確認流程正確，再換成目標模型進行實驗。

### 5.1 Debug 模型

初期請先支援使用小模型進行 debug，例如：

```text
gpt2
TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

使用小模型的目的：

1. 確認 attention weights 可以正確收集。
2. 確認 token-level importance 可以正確轉成 chunk-level importance。
3. 確認 CSV 和圖表可以正確輸出。
4. 降低 GPU memory 壓力，方便快速除錯。

### 5.2 目標模型

後續目標模型為：

```text
meta-llama/Llama-2-7b-hf
facebook/opt-6.7b
```

在 RTX 5080 16GB 上，LLaMA-2 7B FP16 的 full-KV inference 可能只能支援約 1K～2K context length。因此本實驗不要求一開始跑 8K、16K 或更長 context。

對 LLaMA-2 7B 的建議初始設定：

```text
context length: 1024 / 1536 / 2048
sink_size: 8
recent_window: 256 或 512
chunk_size: 128 或 256
max_new_tokens: 32
```

只要 context 可以被切分成 Sink KV、Cold KV 和 Recent KV 三個區域，就可以進行本實驗。

另外，請加入 max_input_tokens 控制 prompt 長度，例如預設為 2048 tokens。
若 context + query tokenized 後超過 max_input_tokens，程式需要明確截斷，並輸出 warning。

截斷時需要注意：
1. 對 local request，應盡量保留最近上下文，避免把最後段落截掉。
2. 對 long_range request，應盡量保留 early / middle evidence，避免把答案線索截掉。
3. 若 sample 被截斷，請在輸出結果或 log 中記錄 sample_id 與截斷前後 token 長度。

### 5.3 Attention output 注意事項

若要取得 attention weights，模型推論時需要開啟：

```python
output_attentions=True
use_cache=True
```

對 LLaMA 類模型，可能需要使用 eager attention implementation 才能回傳 attention weights，例如：

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cuda",
    attn_implementation="eager"
)
```

若模型或環境不支援 `attn_implementation="eager"`，請在程式中加入 fallback，讓小模型仍可正常執行。

### 5.4 Manual decode loop

本實驗需要逐步收集每個 decode step 的 attention weights，因此不要直接使用 `model.generate()`。
在每個 decode step，只使用當前 newly generated token / current query token 對所有 past KV positions 的 attention weights。
若 attention tensor shape 為 [batch, heads, query_len, key_len]，請取 query 維度最後一個位置，即 attention[:, :, -1, :]。

請使用 manual decode loop：

```text
1. 對 prompt 做 tokenize。
2. 若 prompt token 數超過 max_input_tokens，依照 request type 進行截斷，並輸出 warning。
3. 執行一次 prefill，取得 past_key_values。
4. 每次只輸入最新 token 和 past_key_values。
5. 取得 logits、attentions 和更新後的 past_key_values。
6. 使用 greedy decoding 選出下一個 token。
7. 每個 decode step 都計算 attention-based KV importance。

```

## 6. 程式需求與檔案結構

請實作主要程式：

```text
analyze_kv_importance.py
```

此程式負責執行 KV importance analysis，包含模型載入、資料讀取、manual decode、attention collection、importance 計算、CSV 輸出與圖表產生。

建議資料夾結構如下：

```text
exp1_kv_importance/
├── EXP1_kv_importance.md
├── analyze_kv_importance.py
├── requirements.txt
├── data/
│   └── samples.jsonl
└── outputs/
    └── kv_importance/
        ├── chunk_importance.csv
        ├── topk_chunks.csv
        ├── stability_metrics.csv
        ├── attention_mass.csv
        ├── chunk_importance_heatmap.png
        ├── hot_set_overlap.png
        └── attention_mass_distribution.png
```

請優先確保程式可以正確執行與輸出結果，不需要先做效能最佳化。

程式應該包含以下功能：

1. 讀取 Hugging Face causal language model。
2. 讀取 `data/samples.jsonl`。
3. 將 `context` 和 `query` 組合成 prompt。
4. 執行 prefill 並取得 `past_key_values`。
5. 使用 manual decode loop 逐步生成 tokens。
6. 在每個 decode step 收集 attention weights。
7. 將 attention weights 聚合成 token-level importance。
8. 將 token-level importance 轉換成 chunk-level importance。
9. 計算 Hot / Cold stability metrics。
10. 輸出 CSV 檔案。
11. 輸出分析圖表。

## 7. Command-line Arguments

請讓 `analyze_kv_importance.py` 支援以下 command-line arguments：

```bash
python analyze_kv_importance.py \
  --model-name gpt2 \
  --input data/samples.jsonl \
  --output-dir outputs/kv_importance \
  --chunk-size 128 \
  --sink-size 8 \
  --recent-window 512 \
  --top-k 5 \
  --max-input-tokens 2048 \
  --max-new-tokens 32 \
  --device cuda
```

參數說明：

| 參數                 | 預設值                     | 說明                               |
| ------------------ | ----------------------- | -------------------------------- |
| `--model-name`     | `gpt2`                  | Hugging Face model name          |
| `--input`          | `data/samples.jsonl`    | 輸入 sample 檔案                     |
| `--output-dir`     | `outputs/kv_importance` | 輸出 CSV 和圖片的資料夾                   |
| `--chunk-size`     | `128`                   | 每個 KV chunk 包含多少 token positions |
| `--sink-size`      | `8`                     | Sink KV 的 token 數量               |
| `--recent-window`  | `512`                   | Recent KV window 大小              |
| `--top-k`          | `5`                     | 每個 decode step 要觀察前幾個最重要 chunks  |
| `--max-new-tokens` | `32`                    | 每筆 sample 最多 decode 幾個新 tokens   |
| `--device`         | `cuda`                  | 執行裝置，可為 `cuda` 或 `cpu`           |
| `--max-input-tokens` | `2048` | 每筆 sample 的 prompt 最大 token 長度，避免 context 超過模型可處理長度 |

如果 `--device cuda` 但 GPU 不可用，程式應該顯示清楚的錯誤訊息，或自動 fallback 到 CPU。

執行時，程式需要自動建立 output directory。若輸出資料夾不存在，請自動建立。

## 8. 實驗指標

本實驗需要計算以下指標，用來觀察 Hot / Cold KV 的重要程度是否穩定。

---

### 8.1 Top-k hot chunks

對每個 decode step，根據 `chunk_importance` 找出 importance 最高的前 `top_k` 個 chunks。

輸出欄位應包含：

```text
sample_id, step, topk_rank, chunk_id, importance
```

用途：

```text
觀察每個 decode step 中，哪些 KV chunks 最重要。
```

---

### 8.2 Hot set overlap

比較相鄰 decode steps 的 top-k hot chunks 是否相同。

使用 Jaccard similarity：

```text
hot_set_overlap[t]
= |TopK(t) ∩ TopK(t-1)| / |TopK(t) ∪ TopK(t-1)|
```

其中：

```text
TopK(t) = decode step t 的 top-k hot chunks 集合
```

用途：

```text
overlap 越高，代表 hot chunks 越穩定；
overlap 越低，代表 hot chunks 變動越大。
```

---

### 8.3 Hotness churn rate

Hotness churn rate 定義為：

```text
churn_rate[t] = 1 - hot_set_overlap[t]
```

用途：

```text
churn_rate 越高，代表重要 chunks 隨 decode step 改變越頻繁。
```

---

### 8.4 Rank correlation

計算相鄰 decode steps 的 chunk importance ranking 之間的 Spearman rank correlation。

用途：

```text
觀察整體 chunk importance 排名是否穩定。
```

如果 decode step 數量不足，或 chunk 數量不足以計算 Spearman correlation，請輸出空值或 NaN。

---

### 8.5 Attention mass distribution

每個 decode step 都需要計算 attention mass 分布在以下三個區域的比例：

```text
Sink KV
Recent KV
Cold KV
```

定義如下：

```text
Sink KV   = 前 sink_size 個 token positions
Recent KV = decode step 當下最近 recent_window 個 token positions
Cold KV   = Sink KV 和 Recent KV 中間的 token positions
```

輸出欄位應包含：

```text
sample_id, step, sink_mass, recent_mass, cold_mass
```

用途：

```text
觀察 attention 是否主要集中在 Sink / Recent，
以及 Cold KV 是否在某些 request 或 decode step 中變重要。
```

## 9. 輸出檔案格式

程式執行後，請在 `--output-dir` 指定的資料夾中輸出 CSV 檔案與圖表。

預設輸出資料夾為：

```text
outputs/kv_importance/
```

---

### 9.1 `chunk_importance.csv`

此檔案記錄每個 sample、每個 decode step、每個 chunk 的 importance。

欄位：

```text
sample_id, step, chunk_id, token_start, token_end, importance
```

欄位說明：

| 欄位            | 說明                                            |
| ------------- | --------------------------------------------- |
| `sample_id`   | sample 的 id                                   |
| `step`        | decode step 編號，從 0 開始                         |
| `chunk_id`    | chunk 編號                                      |
| `token_start` | 此 chunk 起始 token position                     |
| `token_end`   | 此 chunk 結束 token position                     |
| `importance`  | 此 chunk 在該 decode step 的 attention importance |

---

### 9.2 `topk_chunks.csv`

此檔案記錄每個 decode step 中 importance 最高的 top-k chunks。

欄位：

```text
sample_id, step, topk_rank, chunk_id, importance
```

欄位說明：

| 欄位           | 說明                   |
| ------------ | -------------------- |
| `sample_id`  | sample 的 id          |
| `step`       | decode step 編號       |
| `topk_rank`  | top-k 排名，從 1 開始      |
| `chunk_id`   | 該排名對應的 chunk 編號      |
| `importance` | 該 chunk 的 importance |

---

### 9.3 `stability_metrics.csv`

此檔案記錄 hot chunks 在 decode steps 之間的穩定程度。

欄位：

```text
sample_id, step, hot_set_overlap, churn_rate, spearman_rank_corr
```

欄位說明：

| 欄位                   | 說明                                                                |
| -------------------- | ----------------------------------------------------------------- |
| `sample_id`          | sample 的 id                                                       |
| `step`               | decode step 編號                                                    |
| `hot_set_overlap`    | 當前 step 與前一個 step 的 top-k hot chunks Jaccard overlap              |
| `churn_rate`         | `1 - hot_set_overlap`                                             |
| `spearman_rank_corr` | 當前 step 與前一個 step 的 chunk importance ranking Spearman correlation |

若 `step = 0`，因為沒有前一個 step 可以比較，這些欄位可以輸出空值或 NaN。

---

### 9.4 `attention_mass.csv`

此檔案記錄每個 decode step 的 attention mass 分布。

欄位：

```text
sample_id, step, sink_mass, recent_mass, cold_mass
```

欄位說明：

| 欄位            | 說明                                |
| ------------- | --------------------------------- |
| `sample_id`   | sample 的 id                       |
| `step`        | decode step 編號                    |
| `sink_mass`   | attention mass 落在 Sink KV 區域的比例   |
| `recent_mass` | attention mass 落在 Recent KV 區域的比例 |
| `cold_mass`   | attention mass 落在 Cold KV 區域的比例   |

理想情況下：

```text
sink_mass + recent_mass + cold_mass ≈ 1.0
```

若因為浮點數誤差導致非常小的偏差，可以接受。
Recent KV 不應與 Sink KV 重疊。
在 decode step t，令 total_len 為目前可 attention 的 token 數。
sink_range = [0, min(sink_size, total_len)]
recent_start = max(sink_size, total_len - recent_window)
recent_range = [recent_start, total_len]
cold_range = [sink_size, recent_start]
若 recent_start <= sink_size，則 Cold KV 為空。

---

## 10. 圖表輸出

請至少產生以下三張圖。

---

### 10.1 `chunk_importance_heatmap.png`

用途：

```text
觀察不同 chunks 在不同 decode steps 的 importance 變化。
```

圖表設定：

| 項目     | 內容               |
| ------ | ---------------- |
| X-axis | decode step      |
| Y-axis | chunk_id         |
| Color  | chunk importance |

---

### 10.2 `hot_set_overlap.png`

用途：

```text
觀察 top-k hot chunks 是否會隨 decode step 劇烈改變。
```

圖表設定：

| 項目     | 內容              |
| ------ | --------------- |
| X-axis | decode step     |
| Y-axis | hot_set_overlap |

---

### 10.3 `attention_mass_distribution.png`

用途：

```text
觀察 attention mass 在 Sink KV、Recent KV、Cold KV 之間的分布。
```

圖表設定：

| 項目   | 內容                        |
| ------ | ------------------------- |
| X-axis | decode step               |
| Y-axis | attention mass            |
| Lines  | Sink KV、Recent KV、Cold KV |

## 11. 執行範例

請先建立 Python 環境並安裝必要套件。

建議 `requirements.txt` 至少包含：

```text
torch
transformers
accelerate
pandas
numpy
matplotlib
scipy
```

執行範例一：先用 `gpt2` debug

```bash
python analyze_kv_importance.py \
  --model-name gpt2 \
  --input data/samples.jsonl \
  --output-dir outputs/kv_importance_gpt2 \
  --chunk-size 64 \
  --sink-size 4 \
  --recent-window 128 \
  --top-k 3 \
  --max-new-tokens 16 \
  --device cuda
```

執行範例二：使用較大的模型測試

```bash
python analyze_kv_importance.py \
  --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --input data/samples.jsonl \
  --output-dir outputs/kv_importance_tinyllama \
  --chunk-size 128 \
  --sink-size 8 \
  --recent-window 512 \
  --top-k 5 \
  --max-new-tokens 32 \
  --device cuda
```

執行範例三：後續目標模型 LLaMA-2 7B

```bash
python analyze_kv_importance.py \
  --model-name meta-llama/Llama-2-7b-hf \
  --input data/samples.jsonl \
  --output-dir outputs/kv_importance_llama2_7b \
  --chunk-size 128 \
  --sink-size 8 \
  --recent-window 512 \
  --top-k 5 \
  --max-new-tokens 32 \
  --device cuda
```

注意：LLaMA-2 7B 在 RTX 5080 16GB 上可能會有 VRAM 壓力，因此請先用 `gpt2` 或 TinyLlama 確認程式流程正確，再嘗試 LLaMA-2 7B。

---

## 12. 完成標準

本實驗程式完成後，至少需要滿足以下條件：

1. 可以成功讀取 `data/samples.jsonl`。
2. 可以載入至少一個 Hugging Face causal language model。
3. 可以使用 manual decode loop 逐步生成 tokens。
4. 可以在每個 decode step 收集 attention weights。
5. 可以將 attention weights 聚合成 token-level importance。
6. 可以將 token-level importance 轉換成 chunk-level importance。
7. 可以輸出以下 CSV：

   * `chunk_importance.csv`
   * `topk_chunks.csv`
   * `stability_metrics.csv`
   * `attention_mass.csv`
8. 可以輸出至少以下圖表：

   * `chunk_importance_heatmap.png`
   * `hot_set_overlap.png`
   * `attention_mass_distribution.png`
9. 程式支援以下可調參數：

   * `--model-name`
   * `--input`
   * `--output-dir`
   * `--chunk-size`
   * `--sink-size`
   * `--recent-window`
   * `--top-k`
   * `--max-new-tokens`
   * `--device`
   * `--max-input-tokens`
10. 程式碼需保持清楚、可讀，優先確保 correctness，不需要先做效能最佳化。

## 13. 結果解讀方式

實驗完成後，請根據輸出的 CSV 和圖表觀察以下問題。

---

### 13.1 Hot chunks 是否穩定？

觀察：

```text
topk_chunks.csv
stability_metrics.csv
hot_set_overlap.png
chunk_importance_heatmap.png
```

如果 `hot_set_overlap` 長期偏高，代表相鄰 decode steps 的 top-k hot chunks 大致相同，Hot / Cold KV 可能相對穩定。

如果 `hot_set_overlap` 經常偏低，或 `churn_rate` 偏高，代表 hot chunks 會隨 decode step 改變，可能需要 dynamic hotness tracking 或 request-aware Cold-KV retrieval。

---

### 13.2 Attention 是否主要集中在 Sink / Recent KV？

觀察：

```text
attention_mass.csv
attention_mass_distribution.png
```

如果大部分 attention mass 都集中在 Sink KV 和 Recent KV，代表固定保留 Sink + Recent KV 在 GPU 可能是合理的。

如果 Cold KV 在某些 request 或某些 decode steps 中佔有明顯 attention mass，代表 Cold KV 不能完全丟棄，未來可能需要 selective retrieval。

---

### 13.3 Local request 和 Long-range request 是否不同？

比較 `type = local` 和 `type = long_range` 的 samples。

預期可能觀察到：

```text
local request:
attention 較集中在 Recent KV

long_range request:
某些 early / middle Cold KV chunks 可能變重要
```

如果兩者的 attention mass distribution 或 hot chunk pattern 明顯不同，代表 request type 會影響 KV importance，支援 request-aware KV retrieval 的設計方向。

---

### 13.4 Fixed Hot/Cold placement 是否足夠？

如果實驗結果顯示：

```text
Sink + Recent 長期佔據大多數 attention mass
Top-k hot chunks 變動不大
Cold KV 很少被注意
```

則可以說固定式 Hot/Cold placement 可能已經足夠。

如果實驗結果顯示：

```text
Cold KV 在 long-range request 中會變重要
Top-k hot chunks 會隨 decode step 改變
Cold chunks 偶爾出現高 importance
```

則代表固定式 Hot/Cold placement 不夠，系統需要額外的 dynamic / request-aware retrieval policy。

---

## 14. 最終研究問題對應

本實驗主要用來回答老師提出的問題：

```text
Hot / Cold KV 是否是固定的？
還是 KV importance 會隨時間、decode step 或 request type 改變？
```

實驗結果將用來決定後續系統設計：

1. 如果 Hot / Cold KV 穩定，則 Sink + Recent 固定保留 GPU、Cold KV offload 到外部記憶體可能合理。
2. 如果 Hot / Cold KV 會變動，則 CXL memory 不應該只是被動 offload tier，而應該支援 Cold KV metadata、hotness tracking，以及 selective Cold-KV retrieval。

