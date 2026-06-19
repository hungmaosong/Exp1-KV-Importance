# GPT-2 Pilot Experiment Summary

This report summarizes the GPT-2 synthetic pilot run for Experiment 1: KV Importance / Hot-Cold Stability Analysis.

The goal of this report is to organize pilot observations only. It does not make strong research conclusions. Phrasing such as "observed in this GPT-2 synthetic pilot setting" should be read as scoped to the current synthetic dataset, GPT-2 model, and decoding configuration.

## 1. Experiment Setting

Main analysis script:

```text
analyze_kv_importance.py
```

Post-processing script:

```text
plot_pilot_summary.py
```

Input dataset:

```text
data/samples_pilot_40.jsonl
```

Main output directory:

```text
outputs/kv_importance_gpt2_pilot_40/
```

Run configuration:

```text
model_name: gpt2
chunk_size: 64
sink_size: 4
recent_window: 128
top_k: 3
max_input_tokens: 900
max_new_tokens: 16
device: cuda
```

The main analysis uses full-KV inference and a manual decode loop. At each decode step, the analysis collects the current query token attention with:

```text
attention[:, :, -1, :]
```

Attention is averaged over layers and heads before computing token-level importance, chunk-level importance, Sink / Recent / Cold attention mass, and hot-set stability metrics.

## 2. Dataset Size and Subtype Distribution

The pilot dataset contains 40 synthetic samples.

| subtype | count | analysis type |
|---|---:|---|
| local | 10 | local |
| early_recall | 10 | long_range |
| middle_recall | 10 | long_range |
| comparison | 10 | long_range |

The `early_recall`, `middle_recall`, and `comparison` samples use `type: long_range` so they are compatible with the current main analysis script. Their finer request class is stored in the `subtype` field.

Token length summary from the pilot dataset:

```text
prompt_tokens min / mean / max: 700 / 708.83 / 716
context_tokens min / mean / max: 672 / 680.95 / 688
```

No samples were truncated in this run:

```text
truncation_log.csv: 0 truncated samples out of 40
```

## 3. Output Completeness

The following required CSV files were generated:

| file | rows | columns |
|---|---:|---:|
| `chunk_importance.csv` | 7672 | 6 |
| `topk_chunks.csv` | 1920 | 5 |
| `stability_metrics.csv` | 640 | 5 |
| `attention_mass.csv` | 640 | 5 |

The following required PNG files were generated and verified as readable:

- `chunk_importance_heatmap.png`
- `hot_set_overlap.png`
- `attention_mass_distribution.png`

Additional report-friendly summary outputs were also generated:

- `hot_set_overlap_mean.png`
- `hot_set_overlap_by_subtype.png`
- `attention_mass_by_subtype.png`
- `attention_mass_summary_by_subtype.csv`
- `stability_summary_by_subtype.csv`
- `topk_chunk_frequency_by_subtype.csv`

## 4. Main GPT-2 Pilot Statistics

Overall attention mass summary:

| metric | value |
|---|---:|
| sink_mass mean | 0.2871 |
| recent_mass mean | 0.5340 |
| cold_mass mean | 0.1789 |
| cold_mass min | 0.0717 |
| cold_mass max | 0.2927 |

Chunk summary:

```text
average chunks per sample/step: about 11.99
minimum chunks per sample/step: 11
maximum chunks per sample/step: 12
```

Stability summary:

| metric | min | mean | max |
|---|---:|---:|---:|
| hot_set_overlap | 0.5 | 0.995 | 1.0 |
| churn_rate | 0.0 | 0.005 | 0.5 |

In this GPT-2 synthetic pilot setting, the top-k hot chunks were usually stable across adjacent decode steps, with a small number of drops in hot-set overlap.

## 5. Attention Mass by Subtype

Summary from `attention_mass_summary_by_subtype.csv`:

| subtype | sink_mass_mean | recent_mass_mean | cold_mass_mean | cold_mass_min | cold_mass_max |
|---|---:|---:|---:|---:|---:|
| local | 0.2918 | 0.5669 | 0.1413 | 0.0717 | 0.2208 |
| early_recall | 0.2988 | 0.4821 | 0.2191 | 0.1669 | 0.2666 |
| middle_recall | 0.2784 | 0.5420 | 0.1796 | 0.0897 | 0.2216 |
| comparison | 0.2795 | 0.5450 | 0.1755 | 0.0827 | 0.2927 |

Subtype comparisons:

- Highest `cold_mass_mean`: `early_recall` at 0.2191.
- Highest `recent_mass_mean`: `local` at 0.5669.
- Highest `sink_mass_mean`: `early_recall` at 0.2988.
- Lowest `cold_mass_mean`: `local` at 0.1413.

Pilot observation:

In this GPT-2 synthetic pilot setting, `early_recall` had the highest average Cold KV attention mass among the four subtypes. The `local` samples had the highest average Recent KV attention mass. This matches the synthetic data placement pattern in a qualitative sense, but it should not yet be treated as a general model behavior conclusion.

## 6. Hot Set Overlap by Subtype

Summary from `stability_summary_by_subtype.csv`:

| subtype | hot_set_overlap_mean | hot_set_overlap_min | hot_set_overlap_max | churn_rate_mean | churn_rate_max |
|---|---:|---:|---:|---:|---:|
| local | 0.99 | 0.5 | 1.0 | 0.01 | 0.5 |
| early_recall | 1.00 | 1.0 | 1.0 | 0.00 | 0.0 |
| middle_recall | 0.99 | 0.5 | 1.0 | 0.01 | 0.5 |
| comparison | 1.00 | 1.0 | 1.0 | 0.00 | 0.0 |

Subtype comparisons:

- Lower average `hot_set_overlap`: `local` and `middle_recall`, both around 0.99.
- Highest average `hot_set_overlap`: `early_recall` and `comparison`, both 1.00.
- Nonzero churn was observed in `local` and `middle_recall`.
- `early_recall` and `comparison` had no churn in this summary table.

Pilot observation:

In this GPT-2 synthetic pilot setting, hot chunks were mostly stable across adjacent decode steps. The summary still shows occasional hot-set changes in `local` and `middle_recall`, where `hot_set_overlap_min` drops to 0.5.

## 7. Top-k Chunk Frequency

Summary from `topk_chunk_frequency_by_subtype.csv`.

Most frequent top-k chunks by subtype:

| subtype | most frequent chunk_id values | notes |
|---|---|---|
| local | 0, 10, 11 | chunk 0 and chunk 10 appear in all 160 subtype steps; chunk 11 appears in 156 of 160 steps |
| early_recall | 0, 10, 11 | all three appear in all 160 subtype steps |
| middle_recall | 0, 10, 11 | chunk 0 and chunk 10 appear in all 160 subtype steps; chunk 11 appears in 156 of 160 steps |
| comparison | 0, 10, 11 | all three appear in all 160 subtype steps |

Detailed top-k frequency values:

| subtype | chunk_id | topk_count | frequency | mean_importance |
|---|---:|---:|---:|---:|
| local | 0 | 160 | 0.3333 | 0.3324 |
| local | 10 | 160 | 0.3333 | 0.1564 |
| local | 11 | 156 | 0.3250 | 0.4117 |
| early_recall | 0 | 160 | 0.3333 | 0.4282 |
| early_recall | 10 | 160 | 0.3333 | 0.1724 |
| early_recall | 11 | 160 | 0.3333 | 0.3005 |
| middle_recall | 0 | 160 | 0.3333 | 0.3243 |
| middle_recall | 10 | 160 | 0.3333 | 0.2328 |
| middle_recall | 11 | 156 | 0.3250 | 0.3065 |
| comparison | 0 | 160 | 0.3333 | 0.3264 |
| comparison | 10 | 160 | 0.3333 | 0.1552 |
| comparison | 11 | 160 | 0.3333 | 0.3804 |

Pilot observation:

In this GPT-2 synthetic pilot setting, the most frequent top-k hot chunks are heavily concentrated on chunk 0 and late chunks around chunk 10 and chunk 11. With `chunk_size=64` and prompt lengths around 700 tokens, this roughly corresponds to the Sink / early area and the late prompt / recent area. This is consistent with the high Sink and Recent attention mass, while still leaving nonzero Cold KV mass.

## 8. Initial Pilot Observations

The following observations are scoped to the current GPT-2 synthetic pilot setting:

1. Recent KV received the largest average attention mass overall.
2. Sink KV also received substantial average attention mass.
3. Cold KV mass was consistently nonzero across the pilot run.
4. `early_recall` showed the highest `cold_mass_mean`.
5. `local` showed the highest `recent_mass_mean`.
6. Top-k hot chunks were dominated by chunk 0 and late chunks 10 / 11.
7. Hot-set overlap was mostly high, but a few decode-step transitions showed overlap dropping to 0.5.
8. The cleaner summary plots are more suitable for reporting than the original all-sample `hot_set_overlap.png`.

These are pilot observations only. They should be used to decide the next experiment, not as final evidence about general LLM KV-cache behavior.

## 9. Current Limitations

- The dataset is synthetic and template-generated.
- GPT-2 is a small model and may not represent larger instruction-tuned or LLaMA-style models.
- The generation length is short: `max_new_tokens=16`.
- Attention weight is used as an importance proxy; this is observational and not causal intervention.
- The `early_recall`, `middle_recall`, and `comparison` samples are handled as `type: long_range` by the main script, with finer class stored only in `subtype`.
- The current analysis aggregates attention over layers and heads, so it does not expose layer-specific or head-specific behavior.
- Top-k chunk frequency is influenced by chunk size, prompt template structure, and the fixed Sink / Recent definitions.
- The current pilot does not evaluate whether the generated answer is correct.

## 10. Recommended Next Steps

1. Inspect the report-friendly plots:
   - `outputs/kv_importance_gpt2_pilot_40/attention_mass_by_subtype.png`
   - `outputs/kv_importance_gpt2_pilot_40/hot_set_overlap_mean.png`
   - `outputs/kv_importance_gpt2_pilot_40/hot_set_overlap_by_subtype.png`
2. Review the summary CSVs:
   - `attention_mass_summary_by_subtype.csv`
   - `stability_summary_by_subtype.csv`
   - `topk_chunk_frequency_by_subtype.csv`
3. If the GPT-2 pilot plots look reasonable, run a TinyLlama pilot with the same dataset and comparable settings.
4. Keep `analyze_kv_importance.py` stable for now. Do not modify the main analysis logic unless a correctness issue is found.
5. For TinyLlama, start conservatively with the same `max_new_tokens=16` and monitor memory use before increasing context length or sample count.

---
# GPT解釋

目前這份結果可以這樣理解：**GPT-2 pilot experiment 已經完成，而且結果是合理的；下一步應該換 TinyLlama 做第二個模型驗證。**

---

## 目前 GPT-2 結果代表什麼？

這次 Codex 整理出的報告重點是：

```text
資料：40 筆 synthetic samples
模型：GPT-2
prompt 長度：約 700 tokens
decode 長度：16 tokens
chunk_size：64
平均每筆約 12 個 chunks
```

而且沒有 sample 被截斷，CSV 和 PNG 都完整產生，所以這次已經不是單純 smoke test，而是比較正式的 **GPT-2 pilot experiment**。報告也有強調這只是 pilot observation，還不能下很強的研究結論。

---

## 1. Attention mass 結果怎麼看？

整體平均是：

```text
Recent KV mean = 0.5340
Sink KV mean   = 0.2871
Cold KV mean   = 0.1789
Cold KV max    = 0.2927
```

意思是 GPT-2 在這批資料中生成 token 時：

```text
約 53% attention 看最近區域
約 29% attention 看開頭 / Sink 區域
約 18% attention 看中間 / Cold 區域
```

所以目前不是「Cold KV 沒用」，而是：

```text
Recent KV 最重要；
Sink / early region 也很重要；
Cold KV 較低，但穩定非零。
```

這點對你的研究方向很有用，因為它支持：

```text
Cold KV 可以被視為較冷的區域，但不適合直接丟棄。
```

比較合理的設計會是：

```text
Sink + Recent 放 GPU
Cold KV 放 CXL memory
需要時透過 metadata / hotness tracking / selective retrieval 找回重要 Cold chunks
```

---

## 2. 不同 request subtype 的差異

四類資料結果是：

```text
local：
recent_mass_mean 最高，約 0.5669

early_recall：
cold_mass_mean 最高，約 0.2191
sink_mass_mean 最高，約 0.2988

middle_recall：
cold_mass_mean 約 0.1796

comparison：
cold_mass_mean 約 0.1755
```

這個趨勢滿合理：

```text
local 的答案在最後，所以 Recent KV attention 較高。
early_recall 的線索在前面，所以 Sink / early 區域比較重要。
early_recall 的 Cold KV 也最高，表示長距離資訊可能會讓非 recent 區域變重要。
```

但是要小心一點：因為這是 synthetic data，所以目前只能說：

```text
在 GPT-2 synthetic pilot setting 中觀察到 request subtype 會影響 attention mass 分布。
```

還不能說所有 LLM 都一定如此。

---

## 3. Hot set overlap 結果怎麼看？

整體結果是：

```text
hot_set_overlap mean = 0.995
churn_rate mean = 0.005
```

意思是：

```text
相鄰 decode steps 的 top-k hot chunks 幾乎一樣。
```

也就是在 GPT-2 pilot 裡：

```text
Hot chunks 大多是穩定的，不會每一步一直變。
```

但 local 和 middle_recall 有少數 step 掉到 0.5，代表某些情況還是會有 hot chunk 變動。

所以比較安全的解讀是：

```text
固定式 Hot / Cold placement 可能是一個合理 baseline；
但因為 Cold KV 仍有非零 attention，而且部分 request 有少量 hot chunk 變化，所以仍有必要保留 dynamic / request-aware retrieval 的研究空間。
```

---

## 4. Top-k chunk frequency 怎麼看？

最常出現在 top-k 的 chunks 是：

```text
chunk 0
chunk 10
chunk 11
```

因為 chunk_size = 64，prompt 大約 700 tokens，所以大致可理解成：

```text
chunk 0：開頭 / Sink 附近
chunk 10、11：最後段 / Recent 附近
```

這代表 GPT-2 的 top-k hot chunks 很集中在：

```text
開頭區域 + 最近區域
```

這和 attention mass 的結果一致：Sink / Recent 是主要 hot 區域，Cold 區域有 attention 但比較分散。

---

## 目前可以對老師說什麼？

你現在可以說：

```text
我已經完成 GPT-2 pilot observation。結果顯示，在目前 synthetic pilot setting 中，attention 主要集中在 Recent KV 和 Sink / early region；Cold KV 的平均 attention mass 較低，但穩定非零，約 17.9%。此外，top-k hot chunks 在 decode steps 之間大多穩定，hot_set_overlap 平均約 0.995，表示固定式 Sink + Recent 可以作為合理 baseline。不過不同 request subtype 的 attention mass 分布不同，例如 local 更依賴 Recent KV，而 early_recall 的 Cold KV attention 較高，因此後續仍需要 request-aware / selective Cold-KV retrieval 的設計空間。
```

這樣講很穩，不會下太誇張的結論。

---

## 下一步：跑 TinyLlama

GPT-2 已經完成 pilot，下一步應該用 **TinyLlama/TinyLlama-1.1B-Chat-v1.0** 跑同一份 `samples_pilot_40.jsonl`。

目的不是追求效能，而是確認：

```text
GPT-2 看到的現象，在比較接近現代 decoder-only LLM 的 TinyLlama 上是否也存在。
```

你可以貼給 Codex 這段：

GPT-2 pilot summary 已完成。接下來請進行 TinyLlama pilot experiment。

請注意：

1. 不要修改 `analyze_kv_importance.py` 的主要分析邏輯。
2. 使用同一份資料：

   * `data/samples_pilot_40.jsonl`
3. 請先確認 TinyLlama 是否可以正常載入：

   * `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
4. 請先做一個小型 smoke test，確認模型、attention output、manual decode loop 都正常。
5. 如果 smoke test 成功，再跑完整 pilot。
6. 若發生 CUDA OOM，請先回報錯誤，不要自行大改主程式邏輯。

第一步：TinyLlama smoke test，請輸出到：

`outputs/kv_importance_tinyllama_smoke`

使用指令：

```bash
python analyze_kv_importance.py \
  --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --input data/samples_pilot_40.jsonl \
  --output-dir outputs/kv_importance_tinyllama_smoke \
  --chunk-size 64 \
  --sink-size 4 \
  --recent-window 128 \
  --top-k 3 \
  --max-input-tokens 900 \
  --max-new-tokens 4 \
  --device cuda
```

如果 smoke test 成功，請跑完整 TinyLlama pilot，輸出到：

`outputs/kv_importance_tinyllama_pilot_40`

使用指令：

```bash
python analyze_kv_importance.py \
  --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --input data/samples_pilot_40.jsonl \
  --output-dir outputs/kv_importance_tinyllama_pilot_40 \
  --chunk-size 64 \
  --sink-size 4 \
  --recent-window 128 \
  --top-k 3 \
  --max-input-tokens 900 \
  --max-new-tokens 16 \
  --device cuda
```

完整 TinyLlama pilot 跑完後，請用 `plot_pilot_summary.py` 產生 summary plots / CSV：

```bash
python plot_pilot_summary.py \
  --samples data/samples_pilot_40.jsonl \
  --output-dir outputs/kv_importance_tinyllama_pilot_40
```

請回報以下內容：

1. TinyLlama smoke test 是否成功。
2. 完整 TinyLlama pilot 是否成功。
3. 是否所有 CSV / PNG 都成功產生。
4. 是否有 sample 被截斷。
5. `attention_mass_summary_by_subtype.csv` 的主要數值：

   * 每個 subtype 的 sink_mass_mean
   * recent_mass_mean
   * cold_mass_mean
6. `stability_summary_by_subtype.csv` 的主要數值：

   * hot_set_overlap_mean
   * hot_set_overlap_min
   * churn_rate_mean
7. `topk_chunk_frequency_by_subtype.csv` 中最常出現的 top-k chunks。
8. 請建立一份新報告：

   * `EXPERIMENT_TINYLLAMA_PILOT_SUMMARY.md`

報告格式可以參考 `EXPERIMENT_GPT2_PILOT_SUMMARY.md`，但請明確寫這是 TinyLlama pilot，不要覆蓋 GPT-2 報告。

TinyLlama 跑完後，你再把結果貼給我。下一步我會幫你比較 GPT-2 和 TinyLlama：看哪些現象一致，哪些不一致，這樣就比較能形成可以跟老師討論的實驗結論。
