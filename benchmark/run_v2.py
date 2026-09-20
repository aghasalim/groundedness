"""v2: two domains, eleven languages, seven planted error types plus a control.

    GROQ_API_KEY=... python benchmark/run_v2.py                       # every chat model on Groq
    GEMINI_API_KEY=... python benchmark/run_v2.py --gemini --models gemini-3.8-flash,gemini-2.5-flash

Writes results_v2[-suffix].json and .md. Resumable: a finished (model, lang, domain,
variant) is read from the json and not re-run, so a killed run continues where it stopped.
"""
import argparse, json, os, statistics, sys, time, urllib.error, urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from groundedness import check  # noqa: E402
from run import models_available, OPENROUTER  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def build(cfg):
    """Every (lang, domain, variant) -> (facts, answer, bad)."""
    out = []
    for lang, doms in cfg["languages"].items():
        for dom, t in doms.items():
            if "{days_ret}" not in t["facts"] and dom == "clinic":
                pass
            for var, spec in cfg["variants"].items():
                if var == "days" and "{days_ret}" not in t["answer"]:
                    continue
                slots = dict(cfg["slots"])
                bad = None
                if spec.get("change"):
                    k, v = spec["change"]
                    if "{" + k + "}" not in t["answer"]:
                        continue
                    slots[k] = v
                    bad = v
                facts = t["facts"].format(**cfg["slots"])
                answer = t["answer"].format(**slots)
                if spec.get("append"):
                    sent = t[spec["append"]]
                    answer += sent
                    bad = sent.strip()
                out.append((lang, dom, var, facts, answer, bad))
    return out


def _words(t):
    return {w.strip(".,;:!?()«»\"'").lower() for w in t.split() if len(w.strip(".,;:!?()«»\"'")) > 1}


def matches(bad, claim, var):
    """A changed slot must appear verbatim; an appended sentence is matched by word
    overlap, because judges quote it loosely ('open on Sundays' for 'We are open on Sundays too.')."""
    if var in ("sunday", "extra"):
        b, c = _words(bad), _words(claim)
        return bool(b) and len(b & c) / len(b) >= 0.5
    return bad.lower() in claim.lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models")
    ap.add_argument("--gemini", action="store_true")
    ap.add_argument("--openrouter", action="store_true")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--pace", type=float, default=1.2, help="seconds between calls")
    ap.add_argument("--cases", default="cases_v2.json")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.gemini:
        os.environ["OPENAI_BASE_URL"] = "https://generativelanguage.googleapis.com/v1beta/openai"
        os.environ["OPENAI_API_KEY"] = os.environ["GEMINI_API_KEY"]
        a.suffix = a.suffix or "-gemini"
    if a.openrouter:
        os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
        os.environ["OPENAI_API_KEY"] = os.environ["OPENROUTER_API_KEY"]
        a.suffix = a.suffix or "-openrouter"
        a.models = a.models or ",".join(OPENROUTER)
    cfg = json.load(open(os.path.join(HERE, a.cases), encoding="utf-8"))
    cases = build(cfg)
    models = a.models.split(",") if a.models else models_available()
    path = os.path.join(HERE, a.out or f"results_v2{a.suffix}.json")
    out = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {"cases": len(cases), "models": {}}
    for m in models:
        row = out["models"].setdefault(m, {"calls": {}, "latency": []})
        for lang, dom, var, facts, answer, bad in cases:
            key = f"{lang}/{dom}/{var}"
            if key in row["calls"]:
                continue
            res, err = None, None
            t = time.time()
            for attempt in range(8):
                try:
                    res = check(answer, [facts], m)
                    break
                except urllib.error.HTTPError as e:
                    if e.code in (429, 500, 502, 503) and attempt < 7:
                        time.sleep(10 * (attempt + 1)); t = time.time(); continue
                    err = f"HTTP {e.code}"; break
                except Exception as e:  # noqa: BLE001
                    err = str(e)[:60]; break
            if res is None:
                row["calls"][key] = {"error": err}
            else:
                row["latency"].append(round(time.time() - t, 3))
                if not res.judged:
                    row["calls"][key] = {"unjudged": True}
                elif var == "ok":
                    row["calls"][key] = {"false_alarm": not res.grounded, "unsupported": res.unsupported}
                else:
                    hit = (not res.grounded) and any(matches(bad, u, var) for u in res.unsupported)
                    row["calls"][key] = {"caught": hit, "unsupported": res.unsupported}
                time.sleep(a.pace)
            json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            r = row["calls"][key]
            print(f"{m:28s} {key:22s} {json.dumps({k: v for k, v in r.items() if k != 'unsupported'}, ensure_ascii=False)}", flush=True)
    # summary tables
    variants = [v for v in cfg["variants"] if v != "ok"]
    langs = list(cfg["languages"])
    lines = ["| Model | Errors caught | False alarms | Unjudged/errors | Median latency |", "|---|---:|---:|---:|---:|"]
    summ = {}
    for m, row in out["models"].items():
        c = row["calls"].values()
        caught = sum(1 for x in c if x.get("caught") is True); planted = sum(1 for x in c if "caught" in x or (x.get("unjudged") and True))
        planted = sum(1 for k, x in row["calls"].items() if not k.endswith("/ok") and "error" not in x)
        fa = sum(1 for x in c if x.get("false_alarm") is True); oks = sum(1 for k, x in row["calls"].items() if k.endswith("/ok") and "error" not in x)
        unj = sum(1 for x in c if x.get("unjudged")); errs = sum(1 for x in c if "error" in x)
        lat = statistics.median(row["latency"]) if row["latency"] else None
        summ[m] = (caught, planted, fa, oks, unj, errs, lat)
    for m, (caught, planted, fa, oks, unj, errs, lat) in sorted(summ.items(), key=lambda kv: (-(kv[1][0] / kv[1][1] if kv[1][1] else 0), kv[1][2])):
        lines.append(f"| `{m}` | {caught}/{planted} ({100 * caught / planted if planted else 0:.0f} %) | {fa}/{oks} | {unj}/{errs} | {lat:.2f} s |" if lat else f"| `{m}` | {caught}/{planted} | {fa}/{oks} | {unj}/{errs} | — |")
    lines += ["", "Recall by error type (caught / planted, all languages):", "", "| Model | " + " | ".join(variants) + " |", "|---|" + "---:|" * len(variants)]
    for m, row in out["models"].items():
        cells = []
        for v in variants:
            xs = [x for k, x in row["calls"].items() if k.endswith("/" + v) and "error" not in x]
            cells.append(f"{sum(1 for x in xs if x.get('caught'))}/{len(xs)}")
        lines.append(f"| `{m}` | " + " | ".join(cells) + " |")
    lines += ["", "Recall by language (caught / planted, all error types) · false alarms:", "", "| Model | " + " | ".join(langs) + " |", "|---|" + "---:|" * len(langs)]
    for m, row in out["models"].items():
        cells = []
        for l in langs:
            xs = [x for k, x in row["calls"].items() if k.startswith(l + "/") and not k.endswith("/ok") and "error" not in x]
            fa = sum(1 for k, x in row["calls"].items() if k.startswith(l + "/") and x.get("false_alarm"))
            cells.append(f"{sum(1 for x in xs if x.get('caught'))}/{len(xs)}" + (f" ·{fa}" if fa else ""))
        lines.append(f"| `{m}` | " + " | ".join(cells) + " |")
    open(path.replace(".json", ".md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
