# OPT-6.7B Pilot Experiment Summary

本報告整理 `facebook/opt-6.7b` 在 Experiment 1 synthetic pilot dataset 上的 KV importance / Hot-Cold stability 結果。以下結果屬於 pilot observation，用於觀察較大模型上的 decode-time attention pattern，不代表一般化的 LLM 行為結論。

## 1. 實驗設定

本次使用：

```text
script = analyze_kv_importance_large_model.py
model = facebook/opt-6.7b
input = data/samples_pilot_40.jsonl
output = outputs/kv_importance_opt_6_7b_pilot_40
chunk_size = 64
sink_size = 4
recent_window = 128
top_k = 3
max_input_tokens = 900
max_new_tokens = 16
device = cuda
dtype = float16
attention implementation = eager
```

程式沒有 LLaMA-2 或 OPT 專用 special case。模型與 tokenizer 透過 Hugging Face Auto classes 載入。Prefill 使用完整 prompt、`use_cache=True` 與 `output_attentions=False` 建立 Full-KV cache；後續 manual decode loop 每次輸入目前 token 與完整 `past_key_values`，並使用 `output_attentions=True` 收集：

```text
attention[:, :, -1, :]
```

Decode-time attention 對 heads 與 layers 取 mean，再計算 token/chunk importance、Sink/Recent/Cold mass、top-k hot chunks、Jaccard overlap、churn 與 Spearman rank correlation。每筆 sample 完成後執行 GC、CUDA cache cleanup，並記錄 VRAM。

## 2. Dataset

Synthetic pilot dataset 共 40 筆：

| subtype | samples | evidence placement |
|---|---:|---|
| `local` | 10 | Context 最後段 |
| `early_recall` | 10 | Context 開頭 |
| `middle_recall` | 10 | Context 中間 |
| `comparison` | 10 | 前後兩段皆包含需比較的資訊 |

`local` 使用 `type: local`；其餘 subtype 使用 `type: long_range`。

## 3. 完成狀態與輸出

40/40 samples 全部成功完成，沒有 CUDA OOM、其他 runtime error 或被跳過的 sample。

| subtype | samples | decode rows | min / mean / max steps |
|---|---:|---:|---:|
| `local` | 10 | 160 | 16 / 16 / 16 |
| `early_recall` | 10 | 160 | 16 / 16 / 16 |
| `middle_recall` | 10 | 160 | 16 / 16 / 16 |
| `comparison` | 10 | 160 | 16 / 16 / 16 |
| **Total** | **40** | **640** | **16 / 16 / 16** |

每筆皆完成 step 0 到 step 15，沒有 early EOS。原始 CSV、三張原始 PNG、`memory_usage.csv`、三個 summary CSV 與三張 summary PNG 均成功產生。

正規化檢查：

| check | maximum absolute error |
|---|---:|
| Sink + Recent + Cold mass | `1.94e-07` |
| 每步所有 chunk importance | `1.53e-07` |

誤差屬於正常浮點誤差。

## 4. Truncation

OPT tokenizer 下的 prompt 長度為 700 到 716 tokens，平均 708.825 tokens。沒有 sample 超過 `max_input_tokens=900`，因此：

```text
truncated samples = 0
```

各 subtype token 長度：

| subtype | min | mean | max |
|---|---:|---:|---:|
| `local` | 700 | 712.7 | 716 |
| `early_recall` | 704 | 706.3 | 708 |
| `middle_recall` | 702 | 705.3 | 710 |
| `comparison` | 708 | 711.0 | 714 |

OPT 每個 sample/step 平均形成 11.988 個 chunks，範圍為 11 到 12。

## 5. Overall Attention Mass

640 個 decode rows 的 overall attention mass：

| metric | mean | min | max |
|---|---:|---:|---:|
| `sink_mass` | 0.219021 | 0.160099 | 0.338440 |
| `recent_mass` | 0.534875 | 0.420305 | 0.650678 |
| `cold_mass` | 0.246103 | 0.169921 | 0.323256 |

在這個 OPT-6.7B synthetic pilot setting 中，Recent KV 是平均 attention mass 最大的區域，Cold KV 次之，Sink KV 最低。Cold attention mass 明顯非零，但這仍只是 attention-based pilot observation。

## 6. Attention Mass by Subtype

| subtype | sink mean | recent mean | cold mean | cold min | cold max |
|---|---:|---:|---:|---:|---:|
| `local` | 0.205745 | 0.572680 | 0.221574 | 0.169921 | 0.311112 |
| `early_recall` | 0.243330 | 0.500121 | 0.256550 | 0.184494 | 0.296632 |
| `middle_recall` | 0.211441 | 0.525778 | 0.262782 | 0.183403 | 0.323256 |
| `comparison` | 0.215570 | 0.540923 | 0.243508 | 0.185431 | 0.290613 |

Subtype-level pilot observations：

- `local` 的 `recent_mass_mean` 最高，為 0.572680。
- `early_recall` 的 `sink_mass_mean` 最高，為 0.243330。
- `middle_recall` 的 `cold_mass_mean` 最高，為 0.262782。
- `middle_recall` 也具有最高單一步驟 `cold_mass_max`，為 0.323256。
- 四個 subtype 都以 Recent mass 最大，且 Cold mass 均非零。

## 7. Hot-Set Stability

Overall stability：

| metric | mean | min | max |
|---|---:|---:|---:|
| `hot_set_overlap` | 0.992500 | 0.500000 | 1.000000 |
| `churn_rate` | 0.007500 | 0.000000 | 0.500000 |
| `spearman_rank_corr` | 0.976019 | 0.783217 | 1.000000 |

Subtype-level stability：

| subtype | overlap mean | churn mean | Spearman mean | overlap min | churn max |
|---|---:|---:|---:|---:|---:|
| `local` | 0.996667 | 0.003333 | 0.978993 | 0.500000 | 0.500000 |
| `early_recall` | 0.993333 | 0.006667 | 0.976643 | 0.500000 | 0.500000 |
| `middle_recall` | 0.980000 | 0.020000 | 0.975758 | 0.500000 | 0.500000 |
| `comparison` | 1.000000 | 0.000000 | 0.972681 | 1.000000 | 0.000000 |

在這批 samples 中，top-k hot set 整體高度穩定。`comparison` 的 top-k set 在所有相鄰 steps 都相同；`middle_recall` 的平均 overlap 相對最低、churn 相對最高，但 overlap 仍為 0.98。

## 8. Top-k Chunk Frequency

全部 1,920 個 top-k rows：

| chunk_id | count | share of top-k rows | mean importance |
|---:|---:|---:|---:|
| 0 | 640 | 0.333333 | 0.272872 |
| 10 | 640 | 0.333333 | 0.330349 |
| 11 | 615 | 0.320312 | 0.183596 |
| 5 | 18 | 0.009375 | 0.070834 |
| 9 | 7 | 0.003646 | 0.043981 |

Subtype-level frequency：

| subtype | most frequent chunks |
|---|---|
| `local` | chunk 0 (160), 10 (160), 11 (155), 9 (5) |
| `early_recall` | chunk 0 (160), 10 (160), 11 (158), 9 (2) |
| `middle_recall` | chunk 0 (160), 10 (160), 11 (142), 5 (18) |
| `comparison` | chunk 0 (160), 10 (160), 11 (160) |

Chunk 0 與 chunk 10 在每個 sample/step 都進入 top-k，chunk 11 也在大部分 steps 出現。`middle_recall` 有 18 個 top-k rows 選到 middle chunk 5，但頻率遠低於 chunks 0、10、11。

## 9. VRAM

每筆 sample 完成 GC 與 CUDA cache cleanup 後：

| metric | MiB |
|---|---:|
| allocated after cleanup | 12709.91 |
| reserved after cleanup | 12726.00 |

40 筆 cleanup 後數值完全相同，未觀察到逐筆 GPU memory 累積。

整體 peak：

| metric | MiB | approximately GiB | sample |
|---|---:|---:|---|
| peak allocated | 13223.46 | 12.91 | `local_07` |
| peak reserved | 13402.00 | 13.09 | `local_02` |

## 10. Cross-Model Pilot Comparison

相同設定下的 overall attention mass：

| model | Sink | Recent | Cold |
|---|---:|---:|---:|
| GPT-2 | 0.287135 | 0.534011 | 0.178855 |
| TinyLlama | 0.494890 | 0.346684 | 0.158425 |
| LLaMA-2 7B | 0.606691 | 0.248381 | 0.144928 |
| OPT-6.7B | 0.219021 | 0.534875 | 0.246103 |

相同設定下的 overall stability：

| model | overlap mean | churn mean | Spearman mean |
|---|---:|---:|---:|
| GPT-2 | 0.995000 | 0.005000 | 0.960589 |
| TinyLlama | 0.979651 | 0.020349 | 0.938778 |
| LLaMA-2 7B | 0.949167 | 0.050833 | 0.965729 |
| OPT-6.7B | 0.992500 | 0.007500 | 0.976019 |

在目前 synthetic pilot setting 中，可安全描述的共同趨勢：

1. 四個模型的 Cold attention mass 都非零。
2. 四個模型都在每個 decode row 將 chunk 0 放入 top-k。
3. 四個模型的 hot-set overlap mean 都高於 0.94，hot set 大致穩定但不一定完全固定。
4. `local` 在四個模型中都有最高的 subtype `recent_mass_mean`。
5. 各模型都常選到 late chunks，但 chunk 編號會受到 tokenizer 與 prompt token 長度影響。

可安全描述的模型差異：

1. GPT-2 與 OPT-6.7B 的 attention 較偏 Recent；TinyLlama 與 LLaMA-2 7B 較偏 Sink。
2. OPT-6.7B 的 overall Cold mean 在這四個 pilot runs 中最高；LLaMA-2 7B 最低。
3. Cold mean 最高的 subtype 並不固定：GPT-2 是 `early_recall`、TinyLlama 與 OPT-6.7B 是 `middle_recall`、LLaMA-2 7B 是 `comparison`。
4. GPT-2 與 OPT-6.7B 的 top-k pattern 都主要集中在 chunks 0、10、11；TinyLlama 與 LLaMA-2 7B 則主要集中在 chunk 0 與較晚的 chunks 13、14。
5. GPT-2 與 OPT-6.7B 的 overlap 較高；LLaMA-2 7B 在這批資料中的 churn 相對較高。
6. GPT-2 與 OPT-6.7B 沒有截斷；TinyLlama 與 LLaMA-2 7B 各有 7 筆 comparison samples 截斷。這主要反映 tokenizer 長度差異。

Chunk ID 無法跨 tokenizer 直接一對一比較，因此上述 chunk pattern 應視為相對位置與 frequency observation，而不是完全相同的語意區段。

## 11. Pilot Observations

在 OPT-6.7B synthetic pilot setting 中，Recent KV 是主要 attention 區域，但 Cold KV 平均仍占約 0.246。`middle_recall` 同時具有最高 Cold mean、最高 Cold max，以及相對較高的 hot-set churn；其 top-k 中也偶爾出現 middle chunk 5。這些 observation 與 middle evidence 可能需要 Cold region 的實驗設計方向一致。

目前結果支持將 Sink + Recent 視為 GPU-resident baseline，同時保留 Cold KV 的 metadata、offload 與 selective retrieval 研究空間。不過 attention weight 只是 importance proxy，尚不能據此宣稱 Cold KV retrieval 一定會改善正確率或系統效能。

## 12. Limitations

1. Dataset 是 40 筆 synthetic samples，具有規則化 filler 與 evidence placement。
2. Prompt 長度約 700 到 716 OPT tokens，尚未測試更長 context。
3. Decode 僅有 16 steps，不能代表長生成過程。
4. Attention weight 是 importance proxy，不是 causal proof。
5. 尚未做 Cold KV removal、offload 或 retrieval ablation。
6. 尚未評估 generated answer correctness。
7. Attention 對 layers 與 heads 取 mean，尚未分析 layer/head-specific behavior。
8. Chunk frequency 受到 tokenizer、prompt template、chunk size 與 token boundary 影響。
9. 跨模型的 chunk ID、prompt token 長度與 early EOS 行為不同，不能直接視為完全相同的測量條件。
10. Prefill attention 未回傳；這符合 Experiment 1 的 decode-time 分析目標，但本報告不涵蓋 prefill-time attention。

