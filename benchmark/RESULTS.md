| Model | Recall (planted errors caught) | False alarms (grounded answers flagged) | Median latency | Languages fully caught |
|---|---:|---:|---:|---|
| `openai/gpt-oss-120b` | 22/22 (100 %) | 0/11 | 0.58 s | 11/11: en az ru tr uk kk ar fa hi id vi |
| `openai/gpt-oss-20b` | 22/22 (100 %) | 0/11 | 0.48 s | 11/11: en az ru tr uk kk ar fa hi id vi |
| `qwen/qwen3.8-27b` | 22/22 (100 %) | 0/11 | 0.27 s | 11/11: en az ru tr uk kk ar fa hi id vi |
| `allam-2-7b` | 15/22 (68 %) | 11/11 | 0.25 s | 4/11: en tr ar id |

Per language, columns are: grounded answer wrongly flagged · wrong price caught · invented day caught.

| Model | en | az | ru | tr | uk | kk | ar | fa | hi | id | vi |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `allam-2-7b` | ✗✓✓ | ✗✓✗ | ✗✓✗ | ✗✓✓ | ✗✓✗ | ✗✓✗ | ✗✓✓ | ✗✓✗ | ✗✓✗ | ✗✓✓ | ✗✓✗ |
| `openai/gpt-oss-120b` | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ |
| `openai/gpt-oss-20b` | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ |
| `qwen/qwen3.8-27b` | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ |
