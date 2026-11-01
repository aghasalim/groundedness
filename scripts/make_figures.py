"""Draw the benchmark v3 figures in the README from the saved raw outputs.

Every number is counted from benchmark/results_v3*.json the same way the paper
script counts it: a call with an "error" key is left out, a planted error counts
only if "caught" is true, a grounded answer counts as a false alarm only if
"false_alarm" is true.

    python scripts/make_figures.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmark")
FILES = ("results_v3-siba.json", "results_v3-baselines.json", "results_v3.json")
SHORT = {
    "siba/grounded-multilingual-base": "grounded-multilingual-base (ours)",
    "vectara/HHEM-2.1-Open": "HHEM-2.1-Open",
    "KRLabsOrg/lettucedect-base-modernbert-en-v1": "LettuceDetect-base (en)",
    "openai/gpt-oss-120b": "gpt-oss-120b",
    "openai/gpt-oss-20b": "gpt-oss-20b",
    "qwen/qwen3.8-27b": "qwen3.8-27b",
    "allam-2-7b": "allam-2-7b",
}
TRAINED = {"siba/grounded-multilingual-base": "#1a7f4b",
           "vectara/HHEM-2.1-Open": "#c0392b",
           "KRLabsOrg/lettucedect-base-modernbert-en-v1": "#e08e0b"}
JUDGE = "#4a6fa5"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def load():
    calls = {}
    for f in FILES:
        for m, r in json.load(open(os.path.join(B, f), encoding="utf-8"))["models"].items():
            calls[m] = {k: x for k, x in r["calls"].items() if "error" not in x}
    return calls


def stats(c):
    planted = [x for k, x in c.items() if not k.endswith("/ok")]
    oks = [x for k, x in c.items() if k.endswith("/ok")]
    return dict(caught=sum(bool(x.get("caught")) for x in planted), planted=len(planted),
                fa=sum(bool(x.get("false_alarm")) for x in oks), oks=len(oks),
                langs=len({k.split("/")[0] for k in c}))


def detectors(calls, out):
    """Recall against false-alarm rate, one point per detector."""
    fig, ax = plt.subplots(figsize=(7, 4.4))
    # label positions in data coordinates, picked by hand so the judge cluster near (0, 100) stays readable
    where = {"openai/gpt-oss-120b": (9, 81), "openai/gpt-oss-20b": (9, 88),
             "qwen/qwen3.8-27b": (9, 95), "allam-2-7b": (93, 41),
             "KRLabsOrg/lettucedect-base-modernbert-en-v1": (48, 58)}
    for m, c in calls.items():
        s = stats(c)
        x, y = 100 * s["fa"] / s["oks"], 100 * s["caught"] / s["planted"]
        col = TRAINED.get(m, JUDGE)
        trained = m in TRAINED
        ax.scatter(x, y, s=70, color=col if trained else "white", edgecolor=col, linewidth=1.6, zorder=3)
        label = f"{SHORT[m]}\n{s['caught']}/{s['planted']} caught, {s['fa']}/{s['oks']} false alarms, {s['langs']} lang."
        tx, ty = where.get(m, (x + 2, y))
        ax.annotate(label, (x, y), xytext=(tx, ty), fontsize=7.5, va="center", color=col,
                    ha="right" if tx < x else "left",
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.6, relpos=(1 if tx < x else 0, 0.5)) if m in where else None)
    ax.set_xlim(-3, 100)
    ax.set_ylim(34, 103)
    ax.set_xlabel("grounded answers wrongly flagged (%)")
    ax.set_ylabel("planted errors caught (%)")
    ax.set_title("Benchmark v3: recall against false alarms (top left is best)", loc="left", fontsize=10)
    ax.scatter([], [], s=50, color="0.4", label="trained classifier, all 31 languages")
    ax.scatter([], [], s=50, color="white", edgecolor=JUDGE, linewidth=1.6, label="LLM judge on Groq, languages the quota covered")
    ax.legend(loc="lower left", fontsize=7.5, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def error_types(calls, out):
    """Recall per planted error type for the three trained detectors and one judge."""
    cfg = json.load(open(os.path.join(B, "cases_v3.json"), encoding="utf-8"))
    variants = [v for v in cfg["variants"] if v != "ok"]
    models = list(TRAINED) + ["qwen/qwen3.8-27b"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        s = stats(calls[m])
        rec = []
        for v in variants:
            xs = [x for k, x in calls[m].items() if k.endswith("/" + v)]
            rec.append(100 * sum(bool(x.get("caught")) for x in xs) / len(xs))
        col = TRAINED.get(m, JUDGE)
        ax.bar([j + (i - 1.5) * w for j in range(len(variants))], rec, w, color=col,
               label=f"{SHORT[m]} ({s['langs']} languages)")
    ax.set_xticks(range(len(variants)))
    names = {"price": "price", "hours": "hours", "address": "street no.", "phone": "phone digit",
             "days": "return days", "sunday": "invented\nSunday", "extra": "added\nclaim"}
    ax.set_xticklabels([names.get(v, v) for v in variants])
    ax.set_ylim(0, 105)
    ax.set_ylabel("planted errors caught (%)")
    ax.set_title("Benchmark v3: share of each planted error type caught", loc="left", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left", ncol=2, bbox_to_anchor=(0, -0.36))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    calls = load()
    for fn, name in ((detectors, "detectors_v3.png"), (error_types, "error_types_v3.png")):
        fn(calls, os.path.join(B, name))
        print("->", os.path.join("benchmark", name))
