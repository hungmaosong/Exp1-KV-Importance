# Experiment 1：KV Importance / Hot-Cold Stability Pilot Report

本報告整理 Experiment 1 目前完成的 GPT-2 與 TinyLlama pilot experiment。這些結果應視為 pilot observation，用來判斷後續研究方向與實驗設計是否合理，不代表已經得到一般化的 LLM 行為結論。

## 1. 實驗動機與研究問題

本實驗的核心動機是觀察 LLM 在 decode 過程中，不同 token position 對應的 KV cache importance 是否穩定，以及 Hot / Cold KV 的分界是否可以被簡單固定。

本實驗主要想回答以下問題：

1. Hot / Cold KV 是否是固定的？
2. KV importance 是否會隨 decode step 或 request type 改變？
3. 固定保留 Sink + Recent KV 是否足夠？
4. Cold KV 是否可以直接丟棄，或需要被追蹤與選擇性取回？

目前結果只來自 synthetic pilot setting，包含 GPT-2 與 TinyLlama 兩個小模型。因此，本報告中的說法皆採保守語氣，例如「pilot observation」、「初步觀察」、「目前結果顯示」。這些結果可以支持後續 CXL-aware Cold KV management 的研究方向，但尚不能宣稱已證明一般 LLM 的 KV cache 行為。

## 2. 實驗方法

本實驗使用 full-KV inference 作為觀察基準。也就是說，在觀察 attention distribution 時，不移除任何 past KV cache，先用完整 KV cache 收集每個 decode step 的 attention-based importance。

主要方法如下：

1. 使用 manual decode loop，不使用 `model.generate()`。
2. 對 prompt 做 prefill，取得 `past_key_values`。
3. 每個 decode step 只輸入最新 token 和 `past_key_values`。
4. 每個 decode step 收集目前 query token 對 past KV positions 的 attention：

```text
attention[:, :, -1, :]
```

5. Attention 先對 layers 和 heads 做 mean，得到 token-level importance。
6. 將 token-level importance 依照固定 `chunk_size` 聚合成 chunk-level importance。
7. 每個 decode step 計算 Sink / Recent / Cold attention mass。
8. 每個 decode step 找出 top-k hot chunks。
9. 計算相鄰 decode steps 的 hot set overlap 與 churn rate。

本實驗使用的區域定義如下：

```text
Sink KV   = prompt 開頭 sink_size 個 tokens
Recent KV = decode step 當下最近 recent_window 個 tokens
Cold KV   = Sink KV 與 Recent KV 中間的 tokens
```

在 pilot experiment 中：

```text
chunk_size = 64
sink_size = 4
recent_window = 128
top_k = 3
```

## 3. Dataset 與 Request Type 設計

目前 synthetic pilot dataset 位於：

```text
data/samples_pilot_40.jsonl
```

資料共 40 筆 samples：

| request subtype | samples | 設計目的 |
|---|---:|---|
| `local` | 10 | 答案線索在 context 最後段 |
| `early_recall` | 10 | 答案線索在 context 開頭 |
| `middle_recall` | 10 | 答案線索在 context 中間 |
| `comparison` | 10 | 需要比較前半段與後半段資訊 |

四種 request type 的設計含義如下：

- `local`：答案在最後段，預期 Recent KV 會比較重要。
- `early_recall`：early_recall：答案線索在 context 開頭，預期開頭區域或 chunk 0 會比較重要；Sink KV 則只代表最前面的 sink_size 個 tokens。。
- `middle_recall`：答案在中間，預期 Cold KV 可能會變重要。
- `comparison`：需要比較前後資訊，可能同時需要 Sink / Cold / Recent 區域。

目前主程式只支援 `local` 與 `long_range`。因此，`early_recall`、`middle_recall`、`comparison` 在 JSONL 中使用 `type: long_range`，並透過額外欄位 `subtype` 紀錄細分類型。

## 4. Models and Settings

目前已完成兩個 pilot model：

| model | output directory |
|---|---|
| GPT-2 | `outputs/kv_importance_gpt2_pilot_40/` |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | `outputs/kv_importance_tinyllama_pilot_40/` |

兩個模型使用相同的主要設定：

```text
chunk_size = 64
sink_size = 4
recent_window = 128
top_k = 3
max_input_tokens = 900
max_new_tokens = 16
device = cuda
```

GPT-2 run 完成全部 40 samples，每筆 sample 都跑到 16 decode steps，因此 `attention_mass.csv` 與 `stability_metrics.csv` 各有 640 rows。

TinyLlama run 有兩個需要註明的狀況：

1. TinyLlama tokenizer 下有 7 筆 `comparison` samples 輕微超過 `max_input_tokens=900`，因此被截斷 1 到 6 tokens。
2. TinyLlama 有部分 samples 在 16 decode steps 前產生 EOS，因此 decode rows 少於理論最大值。完整 TinyLlama pilot 的 `attention_mass.csv` 與 `stability_metrics.csv` 各有 556 rows。

這兩點不是程式錯誤，但在比較 GPT-2 與 TinyLlama 時需要保守解讀。

## 5. GPT-2 Pilot Result

GPT-2 pilot 的 overall attention mass：

| metric | value |
|---|---:|
| `sink_mass_mean` | 0.2871 |
| `recent_mass_mean` | 0.5340 |
| `cold_mass_mean` | 0.1789 |

GPT-2 pilot 的 stability summary：

| metric | value |
|---|---:|
| `hot_set_overlap_mean` | 0.9950 |
| `churn_rate_mean` | 0.0050 |

GPT-2 subtype-level attention mass：

| subtype | sink_mass_mean | recent_mass_mean | cold_mass_mean |
|---|---:|---:|---:|
| `local` | 0.2918 | 0.5669 | 0.1413 |
| `early_recall` | 0.2988 | 0.4821 | 0.2191 |
| `middle_recall` | 0.2784 | 0.5420 | 0.1796 |
| `comparison` | 0.2795 | 0.5450 | 0.1755 |

GPT-2 subtype observation：

- `local` 的 `recent_mass_mean` 最高，為 0.5669。
- `early_recall` 的 `cold_mass_mean` 最高，為 0.2191。
- `early_recall` 的 `sink_mass_mean` 最高，為 0.2988。
- top-k hot chunks 主要集中在 chunk 0、10、11。

在 GPT-2 synthetic pilot setting 中，目前結果顯示 Recent KV 是最大的 attention 區域。不過 Cold KV attention mass 並非 0，overall `cold_mass_mean` 為 0.1789，代表中間區域仍然取得一定 attention。這支持一個初步觀察：Cold KV 可能較冷，但不適合在沒有其他機制的情況下直接丟棄。

## 6. TinyLlama Pilot Result

TinyLlama pilot 的 overall attention mass：

| metric | value |
|---|---:|
| `sink_mass_mean` | 0.4949 |
| `recent_mass_mean` | 0.3467 |
| `cold_mass_mean` | 0.1584 |

TinyLlama pilot 的 stability summary：

| metric | value |
|---|---:|
| `hot_set_overlap_mean` | 0.9797 |
| `churn_rate_mean` | 0.0203 |

TinyLlama subtype-level attention mass：

| subtype | sink_mass_mean | recent_mass_mean | cold_mass_mean |
|---|---:|---:|---:|
| `local` | 0.4977 | 0.3731 | 0.1292 |
| `early_recall` | 0.5170 | 0.3176 | 0.1653 |
| `middle_recall` | 0.4746 | 0.3430 | 0.1824 |
| `comparison` | 0.4866 | 0.3557 | 0.1577 |

TinyLlama subtype observation：

- `local` 的 `recent_mass_mean` 最高，為 0.3731。
- `early_recall` 的 `sink_mass_mean` 最高，為 0.5170。
- `middle_recall` 的 `cold_mass_mean` 最高，為 0.1824。
- top-k hot chunks 主要集中在 chunk 0、13、14。
- 在 `middle_recall` 中，chunk 6 經常出現在 top-k hot chunks。

在 TinyLlama synthetic pilot setting 中，目前結果顯示 Sink KV 是最大的 attention 區域。與 GPT-2 不同，TinyLlama 在 `middle_recall` 中呈現較明顯的 middle chunk signal，也就是 chunk 6 經常進入 top-k hot chunks。這是一個值得後續檢查的 pilot observation，但還不能視為一般模型行為。

## 7. GPT-2 vs TinyLlama Comparison

兩個模型的 overall attention mass 比較：

| model | sink_mass_mean | recent_mass_mean | cold_mass_mean |
|---|---:|---:|---:|
| GPT-2 | 0.2871 | 0.5340 | 0.1789 |
| TinyLlama | 0.4949 | 0.3467 | 0.1584 |

兩個模型的 stability 比較：

| model | hot_set_overlap_mean | churn_rate_mean |
|---|---:|---:|
| GPT-2 | 0.9950 | 0.0050 |
| TinyLlama | 0.9797 | 0.0203 |

共同觀察：

1. 兩個模型都顯示 Cold KV attention mass 非零。
2. 兩個模型都常把 chunk 0 放進 top-k hot chunks。
3. 兩個模型都常把 late chunks 放進 top-k hot chunks。
4. 兩個模型的 hot-set overlap 都偏高。
5. `local` 在兩個模型中都有最高 `recent_mass_mean`。
6. `early_recall` 在兩個模型中都有最高 `sink_mass_mean`。

模型差異：

1. GPT-2 attention 較偏 Recent KV。
2. TinyLlama attention 較偏 Sink KV。
3. GPT-2 的最高 Cold subtype 是 `early_recall`。
4. TinyLlama 的最高 Cold subtype 是 `middle_recall`。
5. TinyLlama 的 hot-set churn 比 GPT-2 稍高。
6. TinyLlama 在 `middle_recall` 中出現 chunk 6 的 middle chunk signal。

這些差異可能來自模型架構、tokenizer、prompt chunk boundary、EOS 行為或 synthetic dataset 設計。因此，目前只能視為 pilot observation。

## 8. 初步回答研究問題

### 8.1 Hot / Cold KV 是否固定？

Pilot 中 hot chunks 大多穩定，但不是完全固定。

GPT-2 的 `hot_set_overlap_mean` 為 0.9950，TinyLlama 的 `hot_set_overlap_mean` 為 0.9797，代表相鄰 decode steps 的 top-k hot chunks 大多重疊。不過兩個模型都出現過 `hot_set_overlap_min = 0.5` 的情況，代表 top-k hot chunks 仍會在某些 decode step 發生變動。

因此，目前 pilot observation 是：Hot KV 在這批 synthetic samples 中大致穩定，但不能說完全固定。

### 8.2 Cold KV 是否不重要？

不是。

兩個模型都顯示 Cold KV attention mass 非零：

| model | cold_mass_mean |
|---|---:|
| GPT-2 | 0.1789 |
| TinyLlama | 0.1584 |

此外，TinyLlama 的 `middle_recall` 中 chunk 6 經常出現在 top-k hot chunks，顯示中間區域的 Cold chunk 在某些 request pattern 下可能變得重要。

因此，目前不能把 Cold KV 視為可以直接丟棄的區域。

### 8.3 固定 Sink + Recent 是否足夠？

固定 Sink + Recent 可以作為合理 baseline，因為兩個模型都顯示 Sink / Recent 是主要 attention 區域之一，而且 top-k chunks 也常包含 chunk 0 與 late chunks。

但是，固定 Sink + Recent 不能代表所有重要 KV。Cold KV 在兩個模型中都有非零 attention mass，且 TinyLlama 的 `middle_recall` 顯示 middle Cold chunk 有機會進入 top-k hot chunks。

因此，目前較保守的解讀是：Sink + Recent 可以作為 GPU-resident Hot KV baseline，而 Cold KV 較適合作為 offload / tracked / selectively retrieved 的區域，不適合直接丟棄。

### 8.4 是否支持 CXL-resident Cold KV store？

目前 pilot observation 支持這個方向。

一個合理的後續系統設計方向是：

```text
GPU: keep Sink KV + Recent KV
CXL memory: keep Cold KV
Metadata: maintain chunk-level position / request type / hotness score
Runtime policy: selectively retrieve Cold KV chunks when needed
```

目前結果不代表已證明必須使用 dynamic retrieval，但確實支持「Cold KV 不應只是被動丟棄，而是可以放在 CXL memory 中，並透過 metadata、hotness tracking、request-aware selective retrieval 管理」這個研究方向。

## 9. Limitations

目前 pilot experiment 有以下限制：

1. Dataset 是 synthetic dataset，且帶有 template-generated pattern。
2. 目前只測了 GPT-2 和 TinyLlama 兩個小模型。
3. Attention weight 只是 importance proxy，不是 causal proof。
4. 尚未做 Cold KV removal / ablation experiment。
5. 尚未評估 generated answer correctness。
6. TinyLlama 有 7 筆 samples 被輕微截斷。
7. TinyLlama 有部分 samples 提前產生 EOS，導致 decode rows 少於理論最大值。
8. 目前 analysis 聚合 layers / heads，尚未觀察 layer-specific 或 head-specific behavior。
9. Chunk frequency 會受到 tokenizer、`chunk_size`、prompt template 與 max input length 影響。
10. GPT-2 與 TinyLlama 的 chunk_id 不能完全一對一比較，因為 tokenizer 與 token boundary 不同。

## 10. Next Steps

建議下一步如下：

1. Optional：進行 LLaMA-2 7B smoke test，先確認模型載入、attention output 與 memory 壓力。
2. 建立更自然的 long-context dataset，減少 synthetic template pattern。
3. 增加 samples 數量，觀察 subtype-level pattern 是否穩定。
4. 做 Cold KV removal / ablation experiment，檢查移除特定 Cold chunks 是否真的影響輸出。
5. 比較 Full KV、Sink+Recent only、Attention-based top-k retrieval 三種策略。
6. 未來設計 CXL-resident Cold KV metadata / hotness tracking / selective retrieval。
7. 如要延伸分析，可另外加入 layer/head breakdown，但應作為新功能，不直接改動目前已驗證的主分析邏輯。

## 11. Short Summary for Meeting

這次 Experiment 1 pilot 的目標，是先用 full-KV inference 觀察 decode 過程中不同 KV positions 的 attention importance。初步觀察顯示，GPT-2 比較偏向 Recent KV，而 TinyLlama 比較偏向 Sink KV，但兩個模型的 Cold KV attention mass 都不是 0。Top-k hot chunks 大多穩定，hot-set overlap 偏高，但仍有少數 decode step 出現 churn，因此目前不能說 Hot / Cold KV 是完全固定的。Local request 在兩個模型中都呈現最高 Recent mass，early recall 在兩個模型中都呈現最高 Sink mass，這表示 request type 對 KV importance pattern 可能有影響。TinyLlama 的 middle_recall 中，middle chunk 6 經常進入 top-k hot chunks，這支持後續繼續追蹤 Cold KV 的方向。整體來看，目前 pilot observation 支持把 Sink + Recent 作為 GPU-resident baseline，同時把 Cold KV 放到 CXL memory 中，並保留 metadata、hotness tracking 和 request-aware selective retrieval 的研究空間。不過這些結果仍來自 synthetic pilot dataset，還不能宣稱是 general LLM behavior，後續需要更自然的資料集與 ablation 實驗來驗證。
