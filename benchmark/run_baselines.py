"""The English-trained detectors on the same 154 cases: Vectara HHEM-2.1-Open and LettuceDetect.

    python benchmark/run_baselines.py

HHEM scores factual consistency of (premise=facts, hypothesis=answer) in [0, 1];
below 0.5 counts as "not grounded". LettuceDetect returns hallucinated spans;
any span counts as "not grounded", and a planted error counts as caught when a
span overlaps the planted text. Neither model was trained on anything but
English, which is the point of running them.
"""
import argparse, json, os, statistics, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_v2 import build, _words  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--cases", default="cases_v2.json"); ap.add_argument("--out", default="results_v2-baselines.json"); A = ap.parse_args()
cfg = json.load(open(os.path.join(HERE, A.cases), encoding="utf-8"))
cases = build(cfg)
OUT_P = os.path.join(HERE, A.out)
out = json.load(open(OUT_P, encoding="utf-8")) if os.path.exists(OUT_P) else {"cases": len(cases), "models": {}}


def overlap(span, bad, var):
    if var in ("sunday", "extra"):
        b, s = _words(bad), _words(span)
        return bool(b) and len(b & s) / len(b) >= 0.5
    return bad.lower() in span.lower() or span.lower() in bad.lower()


def record(name, judge):
    row = out["models"].get(name) or {"calls": {}, "latency": []}
    for lang, dom, var, facts, answer, bad in cases:
        if f"{lang}/{dom}/{var}" in row["calls"]:
            continue
        t = time.time()
        flagged, spans = judge(facts, answer)
        row["latency"].append(round(time.time() - t, 3))
        key = f"{lang}/{dom}/{var}"
        if var == "ok":
            row["calls"][key] = {"false_alarm": flagged, "unsupported": spans}
        else:
            # A score-only detector cannot name the claim; flagging the answer is a catch.
            # A span detector must put a span on the planted text.
            named = [x for x in spans if not x.startswith("score=")]
            hit = flagged and (any(overlap(x, bad, var) for x in named) if named else True)
            row["calls"][key] = {"caught": hit, "unsupported": spans}
        print(f"{name:20s} {key:22s} {json.dumps({k: v for k, v in row['calls'][key].items() if k != 'unsupported'})}", flush=True)
    out["models"][name] = row


# --- HHEM-2.1-Open ---
from transformers import AutoModelForSequenceClassification  # noqa: E402
hhem = AutoModelForSequenceClassification.from_pretrained("vectara/hallucination_evaluation_model", trust_remote_code=True)
def hhem_judge(facts, answer):
    score = float(hhem.predict([(facts, answer)])[0])
    return score < 0.5, [f"score={score:.2f}"]
record("vectara/HHEM-2.1-Open", hhem_judge)

# --- LettuceDetect ---
from lettucedetect.models.inference import HallucinationDetector  # noqa: E402
ld = HallucinationDetector(method="transformer", model_path="KRLabsOrg/lettucedect-base-modernbert-en-v1")
def ld_judge(facts, answer):
    spans = ld.predict(context=[facts], answer=answer, output_format="spans")
    texts = [s.get("text", "") for s in spans]
    return bool(spans), texts
record("KRLabsOrg/lettucedect-base-modernbert-en-v1", ld_judge)

json.dump(out, open(OUT_P, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
langs = list(cfg["languages"])
lines = ["| Detector | Errors caught | False alarms | Median latency (CPU) |", "|---|---:|---:|---:|"]
for m, row in out["models"].items():
    c = row["calls"].values()
    caught = sum(1 for x in c if x.get("caught")); planted = sum(1 for k in row["calls"] if not k.endswith("/ok"))
    fa = sum(1 for x in c if x.get("false_alarm")); oks = sum(1 for k in row["calls"] if k.endswith("/ok"))
    lines.append(f"| `{m}` | {caught}/{planted} ({100 * caught / planted:.0f} %) | {fa}/{oks} | {statistics.median(row['latency']):.2f} s |")
lines += ["", "By language (caught / planted · false alarms):", "", "| Detector | " + " | ".join(langs) + " |", "|---|" + "---:|" * len(langs)]
for m, row in out["models"].items():
    cells = []
    for l in langs:
        xs = [x for k, x in row["calls"].items() if k.startswith(l + "/") and not k.endswith("/ok")]
        fa = sum(1 for k, x in row["calls"].items() if k.startswith(l + "/") and x.get("false_alarm"))
        cells.append(f"{sum(1 for x in xs if x.get('caught'))}/{len(xs)} ·{fa}")
    lines.append(f"| `{m}` | " + " | ".join(cells) + " |")
open(OUT_P.replace(".json", ".md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))
