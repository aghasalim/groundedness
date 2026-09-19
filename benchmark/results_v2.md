| Model | Errors caught | False alarms | Unjudged/errors | Median latency |
|---|---:|---:|---:|---:|
| `openai/gpt-oss-120b` | 132/132 (100 %) | 0/22 | 0/0 | 0.70 s |
| `qwen/qwen3.8-27b` | 131/132 (99 %) | 0/22 | 0/0 | 0.36 s |
| `openai/gpt-oss-20b` | 130/132 (98 %) | 1/22 | 0/0 | 0.45 s |
| `allam-2-7b` | 0/0 | 0/0 | 0/154 | — |

Recall by error type (caught / planted, all languages):

| Model | price | hours | address | phone | days | sunday | extra |
|---|---:|---:|---:|---:|---:|---:|---:|
| `allam-2-7b` | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| `openai/gpt-oss-120b` | 22/22 | 11/11 | 22/22 | 22/22 | 11/11 | 22/22 | 22/22 |
| `openai/gpt-oss-20b` | 22/22 | 11/11 | 20/22 | 22/22 | 11/11 | 22/22 | 22/22 |
| `qwen/qwen3.8-27b` | 22/22 | 11/11 | 21/22 | 22/22 | 11/11 | 22/22 | 22/22 |

Recall by language (caught / planted, all error types) · false alarms:

| Model | en | az | ru | tr | uk | kk | ar | fa | hi | id | vi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `allam-2-7b` | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| `openai/gpt-oss-120b` | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 |
| `openai/gpt-oss-20b` | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 11/12 | 12/12 ·1 | 12/12 | 12/12 | 12/12 | 11/12 |
| `qwen/qwen3.8-27b` | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 11/12 | 12/12 | 12/12 |
