| Model | Errors caught | False alarms | Unjudged/errors | Median latency |
|---|---:|---:|---:|---:|
| `gemini-3.8-flash` | 132/132 (100 %) | 0/22 | 0/0 | 2.88 s |
| `gemini-3.5-flash` | 129/132 (98 %) | 0/22 | 0/0 | 5.05 s |
| `gemini-2.5-flash` | 129/132 (98 %) | 0/22 | 0/0 | 5.54 s |

Recall by error type (caught / planted, all languages):

| Model | price | hours | address | phone | days | sunday | extra |
|---|---:|---:|---:|---:|---:|---:|---:|
| `gemini-3.8-flash` | 22/22 | 11/11 | 22/22 | 22/22 | 11/11 | 22/22 | 22/22 |
| `gemini-3.5-flash` | 22/22 | 8/11 | 22/22 | 22/22 | 11/11 | 22/22 | 22/22 |
| `gemini-2.5-flash` | 22/22 | 8/11 | 22/22 | 22/22 | 11/11 | 22/22 | 22/22 |

Recall by language (caught / planted, all error types) · false alarms:

| Model | en | az | ru | tr | uk | kk | ar | fa | hi | id | vi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `gemini-3.8-flash` | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 |
| `gemini-3.5-flash` | 12/12 | 12/12 | 12/12 | 11/12 | 11/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 11/12 |
| `gemini-2.5-flash` | 12/12 | 12/12 | 11/12 | 11/12 | 12/12 | 12/12 | 12/12 | 12/12 | 11/12 | 12/12 | 12/12 |
