"""`python -m groundedness --model M --sources facts.txt < answer.txt`"""
import argparse
import json
import sys

from . import check


def main() -> None:
    ap = argparse.ArgumentParser(prog="groundedness", description="Did the model make this up? Answer on stdin, sources as files.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--sources", nargs="+", required=True, help="text files the answer must be grounded in")
    ap.add_argument("--base-url")
    a = ap.parse_args()
    sources = [open(p, encoding="utf-8").read() for p in a.sources]
    r = check(sys.stdin.read(), sources, a.model, base_url=a.base_url)
    print(json.dumps({"grounded": r.grounded, "score": round(r.score, 3), "unsupported": r.unsupported, "fixed": r.fixed}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
