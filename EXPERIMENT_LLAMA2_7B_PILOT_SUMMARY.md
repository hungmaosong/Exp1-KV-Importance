# LLaMA-2 7B Pilot Experiment Summary

本報告整理 `meta-llama/Llama-2-7b-hf` 在 Experiment 1 synthetic pilot dataset 上的 KV importance / Hot-Cold stability 結果。以下內容屬於 pilot observation，用於檢查較大模型上的實驗相容性與初步 pattern，不代表一般化的 LLM 行為結論。

## 1. 實驗設定

本次使用：

```text
script = analyze_kv_importance_large_model.py
model = meta-llama/Llama-2-7b-hf
input = data/samples_pilot_40.jsonl
output = outputs/kv_importance_llama2_7b_pilot_40
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

新版程式在 prefill 階段仍使用完整 prompt 與 `use_cache=True` 建立 Full-KV cache，但不回傳未被本實驗使用的 prefill attention。後續 manual decode loop 仍逐 token 使用完整 `past_key_values`，並在每個 decode step 收集：

```text
attention[:, :, -1, :]
```

Decode-time attention importance、layer/head mean、chunk aggregation、Sink/Recent/Cold mass、top-k hot chunks、Jaccard overlap、churn 與 Spearman correlation 的定義均未改變。先前 GPT-2 四筆 regression 顯示新版與原版 decode-time CSV 逐位元一致。

## 2. Dataset

Synthetic pilot dataset 共 40 筆：

| subtype | samples | evidence placement |
|---|---:|---|
| `local` | 10 | Context 最後段 |
| `early_recall` | 10 | Context 開頭 |
| `middle_recall` | 10 | Context 中間 |
| `comparison` | 10 | 前後兩段皆包含需比較的資訊 |

`local` 使用 `type: local`；其餘三種 subtype 使用 `type: long_range`。

## 3. 執行完成度與輸出

40 筆 samples 全部成功完成，沒有 CUDA OOM，也沒有跳過失敗 sample。

每筆皆完成 16 個 decode steps，即 step 0 到 step 15：

| subtype | samples | decode rows | min / mean / max steps |
|---|---:|---:|---:|
| `local` | 10 | 160 | 16 / 16 / 16 |
| `early_recall` | 10 | 160 | 16 / 16 / 16 |
| `middle_recall` | 10 | 160 | 16 / 16 / 16 |
| `comparison` | 10 | 160 | 16 / 16 / 16 |
| **Total** | **40** | **640** | **16 / 16 / 16** |

沒有 sample 提前產生 EOS，因此實際 640 decode rows 等於理論最大值。

原始 CSV、三張原始 PNG、`memory_usage.csv`，以及三個 summary CSV 和三張 summary PNG 均成功產生。Attention mass 每步加總的最大誤差為 `2.09e-07`，chunk importance 每步加總的最大誤差為 `3.25e-07`，屬正常浮點誤差。

## 4. Truncation

共有 7 筆 samples 被截斷，全部屬於 `comparison`：

| sample | original tokens | final tokens |
|---|---:|---:|
| `comparison_01` | 901 | 900 |
| `comparison_02` | 904 | 900 |
| `comparison_05` | 901 | 900 |
| `comparison_06` | 905 | 900 |
| `comparison_08` | 901 | 900 |
| `comparison_09` | 901 | 900 |
| `comparison_10` | 906 | 900 |

截斷策略均為：

```text
long_range:preserve_prefix_middle_tail
```

其餘 33 筆沒有截斷。被截斷的 comparison samples 僅超出 1 到 6 tokens，但在解讀 comparison subtype 時仍應保留這項限制。

## 5. Overall Attention Mass

640 個 decode rows 的 overall mean：

| metric | mean |
|---|---:|
| `sink_mass` | 0.606691 |
| `recent_mass` | 0.248381 |
| `cold_mass` | 0.144928 |

在這個 LLaMA-2 7B synthetic pilot setting 中，Sink KV 是平均 attention mass 最大的區域，Recent KV 次之，Cold KV 最低但並非 0。

## 6. Attention Mass by Subtype

| subtype | sink mean | recent mean | cold mean | cold min | cold max |
|---|---:|---:|---:|---:|---:|
| `local` | 0.609995 | 0.279607 | 0.110398 | 0.077420 | 0.162591 |
| `early_recall` | 0.604233 | 0.234014 | 0.161753 | 0.096785 | 0.243361 |
| `middle_recall` | 0.627212 | 0.230013 | 0.142775 | 0.084148 | 0.200485 |
| `comparison` | 0.585325 | 0.249890 | 0.164784 | 0.105083 | 0.205977 |

Subtype-level pilot observations：

- `middle_recall` 的 `sink_mass_mean` 最高，為 0.627212。
- `local` 的 `recent_mass_mean` 最高，為 0.279607。
- `comparison` 的 `cold_mass_mean` 最高，為 0.164784。
- `early_recall` 的單一步驟 `cold_mass_max` 最高，為 0.243361。
- 四個 subtype 的 Cold attention mass 都明顯非零。

## 7. Hot-Set Stability

Overall stability：

| metric | value |
|---|---:|
| `hot_set_overlap_mean` | 0.949167 |
| `churn_rate_mean` | 0.050833 |
| `spearman_rank_corr_mean` | 0.965729 |
| `hot_set_overlap_min` | 0.500000 |
| `churn_rate_max` | 0.500000 |

Subtype-level stability：

| subtype | overlap mean | overlap min | overlap max | churn mean | churn max |
|---|---:|---:|---:|---:|---:|
| `local` | 0.970000 | 0.500000 | 1.000000 | 0.030000 | 0.500000 |
| `early_recall` | 0.920000 | 0.500000 | 1.000000 | 0.080000 | 0.500000 |
| `middle_recall` | 0.906667 | 0.500000 | 1.000000 | 0.093333 | 0.500000 |
| `comparison` | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 |

在這批 samples 中，`comparison` 的 top-k hot set 在所有相鄰 decode steps 都相同。`middle_recall` 的平均 overlap 最低、平均 churn 最高，顯示其 hot chunks 相對較常改變。不過整體 overlap 仍偏高，因此較保守的說法是 hot set 大致穩定，但不是所有 subtype 都完全固定。

## 8. Top-k Chunk Frequency

全部 1,920 個 top-k rows 的主要 chunk frequency：

| chunk_id | count | share of top-k rows |
|---:|---:|---:|
| 0 | 640 | 0.333333 |
| 13 | 640 | 0.333333 |
| 14 | 271 | 0.141146 |
| 12 | 136 | 0.070833 |
| 1 | 125 | 0.065104 |
| 6 | 93 | 0.048438 |
| 4 | 8 | 0.004167 |
| 3 | 7 | 0.003646 |

每個 subtype 的主要 hot chunks：

| subtype | most frequent chunks |
|---|---|
| `local` | chunk 0 (160), 13 (160), 14 (111), 12 (49) |
| `early_recall` | chunk 0 (160), 13 (160), 1 (123), 12 (22) |
| `middle_recall` | chunk 0 (160), 13 (160), 6 (93), 12 (65) |
| `comparison` | chunk 0 (160), 13 (160), 14 (160) |

Chunk 0 和 chunk 13 在每個 sample/step 都進入 top-k。`early_recall` 經常選到 early chunk 1，`middle_recall` 經常選到 middle chunk 6，而 `comparison` 的第三個 hot chunk 固定為 late chunk 14。這些 pattern 與 synthetic evidence placement 有方向上的對應，但仍可能受到 tokenizer、prompt template 與 chunk boundary 影響。

## 9. VRAM

每筆 sample 完成 GC 與 `torch.cuda.empty_cache()` 後：

| metric | MiB |
|---|---:|
| allocated after cleanup | 12860.63 |
| reserved after cleanup | 12874.00 |

40 筆的 cleanup 後數值完全相同，未觀察到逐筆累積。

整體 peak：

| metric | MiB | approximately GiB | sample |
|---|---:|---:|---|
| peak allocated | 13545.89 | 13.23 | `comparison_01` |
| peak reserved | 13664.00 | 13.34 | `early_recall_03` |

## 10. Pilot Observation

在 LLaMA-2 7B synthetic pilot setting 中，attention mass 整體明顯偏向 Sink KV，但 Cold KV 在所有 subtype 中都維持非零。`local` 具有最高 Recent mass，`comparison` 具有最高平均 Cold mass，而 `middle_recall` 的 hot-set churn 相對較高。Top-k frequency 也呈現 request pattern 差異：early recall 常出現 chunk 1，middle recall 常出現 chunk 6，comparison 則固定包含 late chunk 14。

這些結果初步支持 Sink + Recent 作為 GPU-resident baseline，但也顯示 Cold region 仍包含會進入 top-k 的 request-dependent chunks。較保守的系統 implication 是 Cold KV 不宜直接丟棄，而可進一步研究 offload、metadata tracking 與 selective retrieval。不過這仍是 pilot observation，不足以推論一般 LLM 行為。

## 11. Limitations

1. Dataset 為 40 筆 synthetic samples，存在規則化 filler 與 evidence placement。
2. 本次只有 LLaMA-2 7B 單一較大模型，尚未與 OPT-6.7B 比較。
3. Prompt 最長為 900 tokens，decode 最長為 16 tokens，尚不代表更長 context 或 generation。
4. 7 筆 comparison samples 被截斷 1 到 6 tokens。
5. Attention weight 是 importance proxy，不是 causal importance proof。
6. 尚未做 Cold KV removal、offload 或 retrieval ablation。
7. 尚未評估 generated answer correctness。
8. Attention 對 layers 和 heads 取 mean，尚未檢查 layer/head-specific behavior。
9. Chunk frequency 會受到 tokenizer、prompt template、`chunk_size=64` 與 token boundary 影響。
10. Prefill attention 未回傳；這符合 decode-time Experiment 1 的分析目標，但本報告不分析 prefill-time attention。

