| Model | Recall (planted errors caught) | False alarms (grounded answers flagged) | Median latency | Languages fully caught |
|---|---:|---:|---:|---|
| `gemini-2.5-flash` | 22/22 (100 %) | 0/11 | 4.25 s | 11/11: en az ru tr uk kk ar fa hi id vi |
| `gemini-2.5-pro` | 21/21 (100 %) | 0/11 | 10.6 s | 10/11: en ru tr uk kk ar fa hi id vi |
| `gemini-3.5-flash` | 22/22 (100 %) | 0/11 | 4.4 s | 11/11: en az ru tr uk kk ar fa hi id vi |
| `gemini-3.8-flash` | 22/22 (100 %) | 0/11 | 2.66 s | 11/11: en az ru tr uk kk ar fa hi id vi |

Per language, columns are: grounded answer wrongly flagged · wrong price caught · invented day caught. `?` = the model returned no usable judgement (counted as a miss), `!` = request error.

| Model | en | az | ru | tr | uk | kk | ar | fa | hi | id | vi |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `gemini-2.5-flash` | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ |
| `gemini-2.5-pro` | ✓✓✓ | ✓!✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ |
| `gemini-3.5-flash` | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ |
| `gemini-3.8-flash` | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ |
