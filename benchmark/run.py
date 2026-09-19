"""Run every model against every language; write results.json and a Markdown table.

    GROQ_API_KEY=... python benchmark/run.py [--models a,b,c]

Per model and language, three answers: one grounded (must pass), two with one
planted error each (must be caught, and the planted text must be among the
unsupported claims). Recall = caught / planted. False alarms = grounded answers
flagged. Latency is the median per call.
"""
import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from groundedness import check  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def models_available():
    base = os.environ.get("OPENAI_BASE_URL") or "https://api.groq.com/openai/v1"
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY")
    req = urllib.request.Request(f"{base}/models", headers={"authorization": f"Bearer {key}", "user-agent": "groundedness-benchmark"})
    ids = [m["id"] for m in json.load(urllib.request.urlopen(req, timeout=20))["data"]]
    skip = ("whisper", "tts", "guard", "embed", "orpheus", "playai", "safeguard", "compound")
    return sorted(i for i in ids if not any(s in i for s in skip))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", help="comma-separated; default: every chat model the endpoint lists")
    a = ap.parse_args()
    cases = {k: v for k, v in json.load(open(os.path.join(HERE, "cases.json"), encoding="utf-8")).items() if not k.startswith("_")}
    models = a.models.split(",") if a.models else models_available()
    out = {"models": {}, "languages": list(cases)}
    for m in models:
        row = {"per_language": {}, "errors": 0}
        caught = planted = alarms = 0
        lat = []
        for lang, c in cases.items():
            r = {"ok_flagged": None, "price": None, "day": None}
            for kind in ("ok", "price", "day"):
                answer, bad = (c[kind], None) if kind == "ok" else c[kind]
                t = time.time()
                res = None
                for attempt in range(6):
                    try:
                        res = check(answer, [c["facts"]], m)
                        break
                    except urllib.error.HTTPError as e:
                        if e.code == 429 and attempt < 5:
                            time.sleep(8 * (attempt + 1))  # free-tier rate limit; wait it out
                            t = time.time()
                            continue
                        row["errors"] += 1
                        r[kind] = f"error: HTTP {e.code}"
                        break
                    except Exception as e:  # noqa: BLE001
                        row["errors"] += 1
                        r[kind] = f"error: {str(e)[:60]}"
                        break
                if res is None:
                    continue
                lat.append(time.time() - t)
                time.sleep(1.2)
                if kind == "ok":
                    r["ok_flagged"] = not res.grounded
                    alarms += int(not res.grounded)
                else:
                    planted += 1
                    hit = (not res.grounded) and any(bad.lower() in u.lower() for u in res.unsupported)
                    caught += int(hit)
                    r[kind] = hit
            row["per_language"][lang] = r
            print(f"{m:34s} {lang}  ok_flagged={r['ok_flagged']}  price={r['price']}  day={r['day']}", flush=True)
        row.update(recall=caught / planted if planted else None, caught=caught, planted=planted, false_alarms=alarms, grounded_cases=len(cases), median_latency_s=round(statistics.median(lat), 2) if lat else None)
        out["models"][m] = row
    json.dump(out, open(os.path.join(HERE, "results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # the table
    lines = ["| Model | Recall (planted errors caught) | False alarms (grounded answers flagged) | Median latency | Languages fully caught |", "|---|---:|---:|---:|---|"]
    for m, row in sorted(out["models"].items(), key=lambda kv: (-(kv[1]["recall"] or 0), kv[1]["false_alarms"])):
        full = [l for l, r in row["per_language"].items() if r["price"] is True and r["day"] is True]
        lines.append(f"| `{m}` | {row['caught']}/{row['planted']} ({(row['recall'] or 0) * 100:.0f} %) | {row['false_alarms']}/{row['grounded_cases']} | {row['median_latency_s']} s | {len(full)}/{len(cases)}: {' '.join(full)} |")
    lines.append("")
    lines.append("Per language, columns are: grounded answer wrongly flagged · wrong price caught · invented day caught.")
    lines.append("")
    hdr = "| Model | " + " | ".join(cases) + " |"
    lines += [hdr, "|---|" + "---|" * len(cases)]
    sym = lambda v: "✓" if v is True else ("✗" if v is False else "!")
    for m, row in out["models"].items():
        cells = [f"{'✗' if r['ok_flagged'] else '✓'}{sym(r['price'])}{sym(r['day'])}" for r in row["per_language"].values()]
        lines.append(f"| `{m}` | " + " | ".join(cells) + " |")
    open(os.path.join(HERE, "RESULTS.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
