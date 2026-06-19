# TinyLlama Pilot Experiment Summary

This report summarizes the TinyLlama pilot run for Experiment 1: KV Importance / Hot-Cold Stability Analysis.

The goal of this report is to organize pilot observations only. It does not make strong research conclusions. Phrasing such as "observed in this TinyLlama synthetic pilot setting" should be read as scoped to the current synthetic dataset, TinyLlama model, and decoding configuration.

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

Smoke test output directory:

```text
outputs/kv_importance_tinyllama_smoke/
```

Full pilot output directory:

```text
outputs/kv_importance_tinyllama_pilot_40/
```

Smoke test configuration:

```text
model_name: TinyLlama/TinyLlama-1.1B-Chat-v1.0
chunk_size: 64
sink_size: 4
recent_window: 128
top_k: 3
max_input_tokens: 900
max_new_tokens: 4
device: cuda
```

Full pilot configuration:

```text
model_name: TinyLlama/TinyLlama-1.1B-Chat-v1.0
chunk_size: 64
sink_size: 4
recent_window: 128
top_k: 3
max_input_tokens: 900
max_new_tokens: 16
device: cuda
```

The main analysis uses full-KV inference and a manual decode loop. At each decode step, it collects current query token attention:

```text
attention[:, :, -1, :]
```

Attention is averaged over layers and heads before computing token-level importance, chunk-level importance, Sink / Recent / Cold attention mass, and hot-set stability metrics.

`analyze_kv_importance.py` was not modified for this TinyLlama run.

## 2. Dataset Size and Subtype Distribution

The pilot dataset is the same synthetic 40-sample dataset used in the GPT-2 pilot.

| subtype | count | analysis type |
|---|---:|---|
| local | 10 | local |
| early_recall | 10 | long_range |
| middle_recall | 10 | long_range |
| comparison | 10 | long_range |

The `early_recall`, `middle_recall`, and `comparison` samples use `type: long_range` so they are compatible with the current main analysis script. Their finer request class is stored in the `subtype` field.

TinyLlama tokenizer length summary from `truncation_log.csv`:

```text
original prompt tokens min / mean / max: 865 / 884.10 / 906
post-truncation prompt tokens min / mean / max: 865 / 883.62 / 900
```

Seven samples were truncated because TinyLlama tokenization made them slightly longer than `max_input_tokens=900`:

| sample_id | original_tokens | truncated_tokens | strategy |
|---|---:|---:|---|
| comparison_01 | 901 | 900 | long_range:preserve_prefix_middle_tail |
| comparison_02 | 904 | 900 | long_range:preserve_prefix_middle_tail |
| comparison_05 | 901 | 900 | long_range:preserve_prefix_middle_tail |
| comparison_06 | 905 | 900 | long_range:preserve_prefix_middle_tail |
| comparison_08 | 901 | 900 | long_range:preserve_prefix_middle_tail |
| comparison_09 | 901 | 900 | long_range:preserve_prefix_middle_tail |
| comparison_10 | 906 | 900 | long_range:preserve_prefix_middle_tail |

## 3. Smoke Test Status

TinyLlama smoke test completed successfully.

Smoke test observations:

- Model loaded successfully on CUDA.
- Attention outputs were returned.
- Manual decode loop completed.
- Required CSV and PNG files were generated.
- Attention mass sums and chunk importance sums were approximately 1.0.
- Seven comparison samples were truncated from 901-906 tokens to 900 tokens.

Smoke test output row counts:

| file | rows | columns |
|---|---:|---:|
| `chunk_importance.csv` | 2286 | 6 |
| `topk_chunks.csv` | 480 | 5 |
| `stability_metrics.csv` | 160 | 5 |
| `attention_mass.csv` | 160 | 5 |
| `truncation_log.csv` | 40 | 7 |

## 4. Full Pilot Output Completeness

The full TinyLlama pilot completed successfully without CUDA OOM.

The following required CSV files were generated:

| file | rows | columns |
|---|---:|---:|
| `chunk_importance.csv` | 7993 | 6 |
| `topk_chunks.csv` | 1668 | 5 |
| `stability_metrics.csv` | 556 | 5 |
| `attention_mass.csv` | 556 | 5 |
| `truncation_log.csv` | 40 | 7 |

The following required PNG files were generated and verified as readable:

- `chunk_importance_heatmap.png`
- `hot_set_overlap.png`
- `attention_mass_distribution.png`

Additional report-friendly summary outputs were generated:

- `hot_set_overlap_mean.png`
- `hot_set_overlap_by_subtype.png`
- `attention_mass_by_subtype.png`
- `attention_mass_summary_by_subtype.csv`
- `stability_summary_by_subtype.csv`
- `topk_chunk_frequency_by_subtype.csv`

The full pilot has 556 decode-step rows rather than 640 because some TinyLlama generations reached EOS before `max_new_tokens=16`.

Decode-step count by subtype:

| subtype | samples | decode rows | min steps/sample | mean steps/sample | max steps/sample |
|---|---:|---:|---:|---:|---:|
| local | 10 | 153 | 13 | 15.3 | 16 |
| early_recall | 10 | 153 | 15 | 15.3 | 16 |
| middle_recall | 10 | 146 | 13 | 14.6 | 16 |
| comparison | 10 | 104 | 7 | 10.4 | 16 |

## 5. Main TinyLlama Pilot Statistics

Overall attention mass summary:

| metric | value |
|---|---:|
| sink_mass mean | 0.4949 |
| recent_mass mean | 0.3467 |
| cold_mass mean | 0.1584 |

Chunk summary:

```text
average chunks per sample/step: about 14.38
minimum chunks per sample/step: 14
maximum chunks per sample/step: 15
```

Stability summary:

| metric | mean |
|---|---:|
| hot_set_overlap | 0.9797 |
| churn_rate | 0.0203 |

In this TinyLlama synthetic pilot setting, the top-k hot chunks were mostly stable across adjacent decode steps, but less perfectly stable than in the GPT-2 pilot summary.

## 6. Attention Mass by Subtype

Summary from `attention_mass_summary_by_subtype.csv`:

| subtype | sink_mass_mean | recent_mass_mean | cold_mass_mean | cold_mass_min | cold_mass_max |
|---|---:|---:|---:|---:|---:|
| local | 0.4977 | 0.3731 | 0.1292 | 0.0993 | 0.1791 |
| early_recall | 0.5170 | 0.3176 | 0.1653 | 0.1011 | 0.2004 |
| middle_recall | 0.4746 | 0.3430 | 0.1824 | 0.1030 | 0.2406 |
| comparison | 0.4866 | 0.3557 | 0.1577 | 0.1053 | 0.2059 |

Subtype comparisons:

- Highest `cold_mass_mean`: `middle_recall` at 0.1824.
- Highest `recent_mass_mean`: `local` at 0.3731.
- Highest `sink_mass_mean`: `early_recall` at 0.5170.
- Lowest `cold_mass_mean`: `local` at 0.1292.

Pilot observation:

In this TinyLlama synthetic pilot setting, Sink KV received the largest average attention mass for every subtype. `middle_recall` had the highest average Cold KV mass, while `local` had the highest average Recent KV mass. These observations are scoped to this synthetic dataset and should not be treated as general TinyLlama behavior without additional runs.

## 7. Hot Set Overlap by Subtype

Summary from `stability_summary_by_subtype.csv`:

| subtype | hot_set_overlap_mean | hot_set_overlap_min | hot_set_overlap_max | churn_rate_mean | churn_rate_max |
|---|---:|---:|---:|---:|---:|
| local | 0.9685 | 0.5 | 1.0 | 0.0315 | 0.5 |
| early_recall | 0.9825 | 0.5 | 1.0 | 0.0175 | 0.5 |
| middle_recall | 0.9743 | 0.5 | 1.0 | 0.0257 | 0.5 |
| comparison | 1.0000 | 1.0 | 1.0 | 0.0000 | 0.0 |

Subtype comparisons:

- Lowest average `hot_set_overlap`: `local` at 0.9685.
- Highest average `hot_set_overlap`: `comparison` at 1.0000.
- Highest `churn_rate_mean`: `local` at 0.0315.
- `comparison` showed no hot-set churn in this summary table.

Pilot observation:

In this TinyLlama synthetic pilot setting, hot chunks were mostly stable, but `local`, `early_recall`, and `middle_recall` each had at least one adjacent-step transition where hot-set overlap dropped to 0.5. The `comparison` subtype was stable in the summarized top-k set, though it also had fewer decode rows due to earlier EOS.

## 8. Top-k Chunk Frequency

Summary from `topk_chunk_frequency_by_subtype.csv`.

Most frequent top-k chunks by subtype:

| subtype | most frequent chunk_id values | notes |
|---|---|---|
| local | 0, 13, 14, 12 | chunk 0 and chunk 13 appear in all 153 top-k step groups; chunk 14 appears often; chunk 12 appears sometimes |
| early_recall | 0, 13, 12 | chunk 0 and chunk 13 appear in all 153 top-k step groups; chunk 12 appears in most groups |
| middle_recall | 0, 13, 6, 12 | chunk 0 and chunk 13 appear in all 146 top-k step groups; chunk 6 appears often |
| comparison | 0, 13, 14 | all three appear in all 104 top-k step groups |

Detailed top-k frequency values:

| subtype | chunk_id | topk_count | frequency | mean_importance |
|---|---:|---:|---:|---:|
| local | 0 | 153 | 0.3333 | 0.5121 |
| local | 13 | 153 | 0.3333 | 0.2551 |
| local | 14 | 104 | 0.2266 | 0.1294 |
| local | 12 | 49 | 0.1068 | 0.0465 |
| early_recall | 0 | 153 | 0.3333 | 0.5652 |
| early_recall | 13 | 153 | 0.3333 | 0.2909 |
| early_recall | 12 | 150 | 0.3268 | 0.0244 |
| early_recall | 1 | 2 | 0.0044 | 0.0226 |
| early_recall | 14 | 1 | 0.0022 | 0.0465 |
| middle_recall | 0 | 146 | 0.3333 | 0.4867 |
| middle_recall | 13 | 146 | 0.3333 | 0.3084 |
| middle_recall | 6 | 139 | 0.3174 | 0.0595 |
| middle_recall | 12 | 7 | 0.0160 | 0.0232 |
| comparison | 0 | 104 | 0.3333 | 0.5032 |
| comparison | 13 | 104 | 0.3333 | 0.1695 |
| comparison | 14 | 104 | 0.3333 | 0.1719 |

Pilot observation:

In this TinyLlama synthetic pilot setting, the most frequent top-k chunks are strongly concentrated around chunk 0 and late chunks around chunk 13 / 14. The `middle_recall` subtype additionally shows chunk 6 frequently in the top-k set, which aligns with the synthetic placement of middle evidence. This is a pilot observation only.

## 9. Initial TinyLlama Pilot Observations

The following observations are scoped to the current TinyLlama synthetic pilot setting:

1. Smoke test and full pilot both completed successfully on CUDA.
2. Seven comparison samples were truncated by 1-6 tokens under TinyLlama tokenization.
3. Some samples stopped before 16 decode steps because EOS was generated.
4. Sink KV received the largest average attention mass overall.
5. Recent KV received the second-largest average attention mass overall.
6. Cold KV mass was consistently nonzero.
7. `middle_recall` showed the highest `cold_mass_mean`.
8. `local` showed the highest `recent_mass_mean`.
9. `early_recall` showed the highest `sink_mass_mean`.
10. Top-k hot chunks were dominated by chunk 0 and late chunks 13 / 14, with chunk 6 appearing frequently for `middle_recall`.

These are pilot observations only. They should be used to decide whether and how to scale the next experiment, not as final evidence about general LLM KV-cache behavior.

## 10. Current Limitations

- The dataset is synthetic and template-generated.
- TinyLlama is still a small model compared with the later target models.
- Generation length is capped at `max_new_tokens=16`, and several samples ended earlier due to EOS.
- Seven samples were slightly truncated because TinyLlama tokenization exceeded 900 input tokens.
- Attention weight is used as an importance proxy; this is observational and not causal intervention.
- The `early_recall`, `middle_recall`, and `comparison` samples are handled as `type: long_range` by the main script, with finer class stored only in `subtype`.
- The current analysis aggregates attention over layers and heads, so it does not expose layer-specific or head-specific behavior.
- Top-k chunk frequency is influenced by chunk size, prompt template structure, tokenizer differences, and fixed Sink / Recent definitions.
- The current pilot does not evaluate whether generated answers are correct.

## 11. Recommended Next Steps

1. Inspect the TinyLlama summary plots:
   - `outputs/kv_importance_tinyllama_pilot_40/attention_mass_by_subtype.png`
   - `outputs/kv_importance_tinyllama_pilot_40/hot_set_overlap_mean.png`
   - `outputs/kv_importance_tinyllama_pilot_40/hot_set_overlap_by_subtype.png`
2. Compare TinyLlama and GPT-2 summary CSVs side by side:
   - `attention_mass_summary_by_subtype.csv`
   - `stability_summary_by_subtype.csv`
   - `topk_chunk_frequency_by_subtype.csv`
3. Consider increasing `max_input_tokens` slightly for TinyLlama if avoiding the 7 small truncations is important.
4. Keep `analyze_kv_importance.py` stable for now. Do not modify the main analysis logic unless a correctness issue is found.
5. Before moving to a larger target model, decide whether early EOS behavior should be accepted as part of the observation or controlled with prompt/data changes.

---
# GPT解釋

可以，這次 TinyLlama 結果比 GPT-2 更有價值，因為 TinyLlama 比 GPT-2 更接近你後面想討論的 decoder-only LLM inference 情境。先講總結：

**TinyLlama pilot 顯示：Hot chunks 大致穩定，但不像 GPT-2 那麼穩；而且 Cold KV 確實有非零 attention，特別是 middle_recall 類型最明顯。**

---

## 1. 這次實驗狀態是否正常？

正常。

TinyLlama smoke test 和 full pilot 都成功，模型可以載入 CUDA、可以回傳 attention、manual decode loop 也能跑完整流程。完整 pilot 沒有 CUDA OOM，必要 CSV、PNG 和 summary outputs 都有成功產生。

不過有兩個小狀況要記得：

```text
1. 有 7 筆 comparison samples 被截斷
2. attention_mass.csv 是 556 rows，不是理論上的 640 rows
```

第一點是因為 TinyLlama tokenizer 算出來的 token 比 GPT-2 多一點，有 7 筆 comparison 從 901–906 tokens 被截到 900。
第二點是因為有些 sample 在生成 16 tokens 前就產生 EOS，所以提前結束；報告中也列出 comparison 平均只生成 10.4 steps，明顯比其他 subtype 少。

這兩點不是程式錯，但之後比較結果時要註明。

---

## 2. Attention Mass 結果怎麼解讀？

整體平均是：

```text
Sink KV mean   = 0.4949
Recent KV mean = 0.3467
Cold KV mean   = 0.1584
```

也就是 TinyLlama 在這批 samples 中，平均大約：

```text
49.5% attention 看 Sink / 開頭區域
34.7% attention 看 Recent 區域
15.8% attention 看 Cold / 中間區域
```

這和 GPT-2 不太一樣。GPT-2 是 Recent KV 最高，但 TinyLlama 是 Sink KV 最高。報告中也明確寫到 TinyLlama 的 Sink KV 是整體最大，Recent 第二，Cold 穩定非零。

這代表：

```text
TinyLlama 對開頭區域 / sink tokens 的依賴更強。
```

所以不能簡單說所有模型都只看 Recent KV。至少在 TinyLlama pilot 中，開頭區域非常重要。

---

## 3. 四種 subtype 怎麼看？

TinyLlama by subtype 的 attention mass 是：

```text
local:         sink 0.4977, recent 0.3731, cold 0.1292
early_recall:  sink 0.5170, recent 0.3176, cold 0.1653
middle_recall: sink 0.4746, recent 0.3430, cold 0.1824
comparison:    sink 0.4866, recent 0.3557, cold 0.1577
```

報告整理的結論是：`middle_recall` 的 cold_mass_mean 最高，`local` 的 recent_mass_mean 最高，`early_recall` 的 sink_mass_mean 最高，`local` 的 cold_mass_mean 最低。

白話解釋如下。

`local` 的答案在最後段，所以 Recent KV 比其他類型更高，這符合預期。但 TinyLlama 的 local 仍然有接近 49.8% Sink attention，表示它仍然很看開頭區域。

`early_recall` 的答案在開頭，所以 sink_mass_mean 最高，這也符合預期。

`middle_recall` 的答案在中間，所以 cold_mass_mean 最高，這是這次 TinyLlama 最重要的觀察。它代表當任務需要中間資訊時，Cold KV 的重要性確實會上升。

`comparison` 需要比較前後資訊，Cold KV 也非零，但這類有 7 筆被截斷、且 EOS 提早比較明顯，所以目前解讀要比其他類型更保守。

---

## 4. Hot Set Overlap 怎麼看？

TinyLlama 的整體穩定性是：

```text
hot_set_overlap mean = 0.9797
churn_rate mean      = 0.0203
```

意思是：

```text
相鄰 decode steps 的 top-k hot chunks 大多相同，但偶爾會變。
```

這比 GPT-2 的 overlap mean 0.995 低一點，代表 TinyLlama 的 hot chunks 沒有 GPT-2 那麼固定。報告也寫到 TinyLlama 的 top-k hot chunks mostly stable，但 less perfectly stable than GPT-2。

分 subtype 來看：

```text
local:         overlap_mean 0.9685
early_recall:  overlap_mean 0.9825
middle_recall: overlap_mean 0.9743
comparison:    overlap_mean 1.0000
```

其中 local 的 overlap 最低、churn 最高；comparison 完全穩定，但它 decode rows 較少，因為比較容易提前 EOS。

所以比較安全的說法是：

```text
TinyLlama 的 hot chunks 大多穩定，但 local、early_recall、middle_recall 都曾出現 hot_set_overlap 掉到 0.5 的情況，代表某些 decode step 的 hot chunks 會變。
```

---

## 5. Top-k Hot Chunks 怎麼看？

TinyLlama 最常出現的 top-k chunks 是：

```text
chunk 0
chunk 13
chunk 14
```

其中 chunk 0 代表開頭 / Sink 附近，chunk 13、14 代表 prompt 後段 / Recent 附近。這表示 TinyLlama 的 top-k hot chunks 主要集中在：

```text
開頭區域 + 最後區域
```

這和 Sink + Recent 佔多數 attention 的結果一致。

但最有意思的是：

```text
middle_recall 額外常出現 chunk 6
```

報告中列出 middle_recall 的 top-k chunks 是 0、13、6、12，其中 chunk 6 出現 139 次，代表中間 chunk 真的常被選進 top-k。

這點很重要，因為它正好對應你的研究問題：

```text
當任務需要中間資訊時，Cold chunk 可能會變成 hot chunk。
```

這比 GPT-2 結果更能支持「Cold KV 需要 selective retrieval」的方向。

---

## 6. 和 GPT-2 相比，TinyLlama 有什麼差異？

可以這樣整理：

| 觀察              | GPT-2              | TinyLlama                             |
| --------------- | ------------------ | ------------------------------------- |
| 最高 attention 區域 | Recent KV          | Sink KV                               |
| Cold KV 是否非零    | 是                  | 是                                     |
| 哪類 Cold KV 最高   | early_recall       | middle_recall                         |
| Hot chunks 穩定性  | 很高，overlap 約 0.995 | 仍高，但較低，overlap 約 0.9797               |
| Top-k chunks    | chunk 0、10、11      | chunk 0、13、14，middle_recall 有 chunk 6 |

最關鍵的差異是：

```text
GPT-2 比較偏 Recent；
TinyLlama 比較偏 Sink；
TinyLlama 的 middle_recall 更明顯把中間 chunk 選進 top-k。
```

所以你可以跟老師說：

```text
不同模型的 attention distribution 不完全相同，但兩個模型都顯示 Cold KV 不是完全沒用。
```

這是比較好的研究敘事。

---

## 目前 TinyLlama 可以得到的安全結論

你可以這樣說：

```text
在 TinyLlama synthetic pilot setting 中，Sink KV 是平均 attention mass 最高的區域，Recent KV 第二，Cold KV 雖然最低但穩定非零。依 subtype 來看，local 的 Recent KV attention 最高，early_recall 的 Sink KV attention 最高，而 middle_recall 的 Cold KV attention 最高。這表示 request type 會影響 KV importance 分布。Top-k hot chunks 大多集中在 chunk 0 和 late chunks 13/14，代表 Sink + Recent 仍是主要 Hot KV，但 middle_recall 中 chunk 6 經常進入 top-k，顯示某些 Cold chunks 會因任務需求變重要。
```

---

## 這對你的研究題目代表什麼？

現在你已經有比較清楚的方向：

```text
固定 Sink + Recent 是合理 baseline，
但不能代表所有重要 KV。
```

因為：

```text
1. Cold KV attention 不是 0
2. middle_recall 的 Cold KV attention 最高
3. middle_recall 的 chunk 6 經常進入 top-k
4. TinyLlama hot chunks 有少量 churn
```

所以這可以支撐你後續設計：

```text
Cold KV 可以 offload 到 CXL memory；
但 CXL memory 不應只是被動放資料，
而應該保留 chunk metadata / hotness score / request-aware retrieval，
在 long-range 或 middle-recall 類任務中取回重要 Cold chunks。
```

---

## 下一步建議

下一步不要馬上跑更大模型。先叫 Codex 做一份 **GPT-2 vs TinyLlama comparison report**，把兩個模型的結果放在一起比較。這樣才會形成一個完整實驗段落。

你可以跟 Codex 說：

```text
請建立一份 GPT-2 vs TinyLlama 的比較報告，不要重新跑模型，也不要修改 analyze_kv_importance.py。

請讀取：
1. EXPERIMENT_GPT2_PILOT_SUMMARY.md
2. EXPERIMENT_TINYLLAMA_PILOT_SUMMARY.md
3. outputs/kv_importance_gpt2_pilot_40/summary CSVs
4. outputs/kv_importance_tinyllama_pilot_40/summary CSVs

請產生：
EXPERIMENT_MODEL_COMPARISON_SUMMARY.md

報告內容請包含：
1. 實驗設定比較
2. GPT-2 與 TinyLlama 的 overall attention mass 比較
3. subtype-level attention mass 比較
4. hot set overlap / churn 比較
5. top-k chunk frequency 比較
6. 共同觀察
7. 模型差異
8. 對 Hot/Cold KV stability 的初步 implication
9. 限制與下一步

請使用保守說法，例如：
“in this synthetic pilot setting”
“pilot observation”
“does not yet imply general LLM behavior”

不要下強結論。
```

完成這份比較報告後，你這個「Hot / Cold KV 是否固定」觀察性實驗就算有一版完整成果了。
