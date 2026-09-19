"""groundedness — did the model make this up?

One call, any OpenAI-compatible model, any language. The judge reads the
answer in its own language, which is what makes it work where every trained
detector (HHEM, LettuceDetect, Lynx) is English-only.

    from groundedness import check
    r = check(answer, sources=[doc], model="qwen/qwen3.8-27b")
    r.unsupported   # claims the sources do not support
    r.fixed         # the answer with those claims removed, same language
    r.score         # 1.0 = fully grounded

No dependencies. Uses OPENAI_BASE_URL / OPENAI_API_KEY (or GROQ_API_KEY).
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field

__version__ = "0.1.2"
__all__ = ["check", "Result", "PROMPT"]

PROMPT = (
    "You check an assistant's answer against the only facts it was allowed to use. Facts:\n{facts}\n\n"
    "List every factual claim in the answer (a price, a date, a time, an address, a phone, a name, "
    "an availability, a rule, a number, a quote) that the facts above do not support. Greetings, offers "
    "to help, and promises to follow up are not claims. Then rewrite the answer in the same language and "
    "tone with the unsupported claims removed, saying plainly that you do not have that information. "
    "If no claim is unsupported, return the answer unchanged. Reply with JSON only: "
    '{{"unsupported": ["..."], "answer": "..."}}'
)


@dataclass
class Result:
    unsupported: list[str] = field(default_factory=list)
    fixed: str = ""
    original: str = ""
    model: str = ""
    raw: str = ""
    judged: bool = True
    """False when the model returned nothing usable: treat as unknown, never as grounded."""

    @property
    def grounded(self) -> bool:
        return self.judged and not self.unsupported

    @property
    def score(self) -> float:
        """1.0 when nothing was unsupported; otherwise 1 minus the share of the answer's text the unsupported claims make up."""
        if not self.unsupported:
            return 1.0
        if not self.original:
            return 0.0
        bad = sum(len(u) for u in self.unsupported)
        return max(0.0, 1.0 - min(1.0, bad / len(self.original)))


def _endpoint() -> tuple[str, str]:
    base = os.environ.get("OPENAI_BASE_URL") or ("https://api.groq.com/openai/v1" if os.environ.get("GROQ_API_KEY") else "https://api.openai.com/v1")
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY") or ""
    return base.rstrip("/"), key


def _facts(sources) -> str:
    if isinstance(sources, str):
        sources = [sources]
    return "\n\n".join(f"[{i + 1}] {s}" for i, s in enumerate(sources) if s and s.strip())


def check(answer: str, sources, model: str, *, base_url: str | None = None, api_key: str | None = None, timeout: float = 30.0, temperature: float = 0.0) -> Result:
    """Judge `answer` against `sources` with `model` over an OpenAI-compatible chat endpoint.

    Never raises on a bad model reply: a judge that cannot be parsed returns the
    answer unchanged with `unsupported=[]` and `raw` set, so a caller can log it.
    Network and auth errors do raise.
    """
    facts = _facts(sources)
    if not facts or not answer or not answer.strip():
        return Result(fixed=answer, original=answer, model=model)
    base, key = _endpoint()
    base = (base_url or base).rstrip("/")
    key = api_key or key
    # Thinking models spend tokens before the first visible character; the
    # budget has to cover that or the reply comes back empty.
    body = {"model": model, "temperature": temperature, "max_tokens": 8000, "messages": [{"role": "system", "content": PROMPT.format(facts=facts)}, {"role": "user", "content": answer}]}
    if "qwen3" in model:
        body["reasoning_format"] = "hidden"
    if "gpt-oss" in model:
        body["reasoning_effort"] = "low"
    headers = {"content-type": "application/json", "authorization": f"Bearer {key}", "user-agent": f"groundedness/{__version__} (+https://github.com/aghasalim/groundedness)"}
    # Small models cap max_tokens below the thinking-model budget; on that
    # specific 400, retry once with the cap the server names (or 2000).
    data = None
    for _attempt in (1, 2):
        req = urllib.request.Request(f"{base}/chat/completions", data=json.dumps(body).encode(), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.load(r)
            break
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace") if e.code == 400 else ""
            m = re.search(r"max_tokens.*?`?(\d{3,6})`?", msg)
            if _attempt == 1 and e.code == 400 and "max_tokens" in msg:
                body["max_tokens"] = min(int(m.group(1)) if m else 2000, body["max_tokens"] - 1)
                continue
            raise
    if data is None:
        return Result(fixed=answer, original=answer, model=model, judged=False)
    raw = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
    if not raw.strip():
        return Result(fixed=answer, original=answer, model=model, raw=raw, judged=False)
    text = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    try:
        j = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        try:
            j = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            j = {}
    if not isinstance(j, dict) or "unsupported" not in j:
        return Result(fixed=answer, original=answer, model=model, raw=raw, judged=False)
    unsupported = [str(x).strip() for x in j.get("unsupported", []) if str(x).strip()]
    fixed = j.get("answer") if isinstance(j.get("answer"), str) else None
    return Result(unsupported=unsupported, fixed=(fixed or answer).strip(), original=answer, model=model, raw=raw)
