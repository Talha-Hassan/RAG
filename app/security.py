"""Input sanitization and prompt-injection defense for user chat messages.

Two layers are applied before a message ever reaches the LLM:
  1. `analyze` / `sanitize_message` - strip hostile unicode, normalize
     whitespace, and score the text against known jailbreak / instruction-
     override / role-delimiter-spoofing patterns. High-risk messages are
     rejected with `InjectionDetected` instead of being silently rewritten,
     since silently mutating a therapy message can change its meaning.
  2. `wrap_for_prompt` - fences the *sanitized* text with an explicit
     delimiter so the calling code never string-concatenates raw user input
     next to the system prompt. Callers must build LLM messages from this
     wrapped form, not the raw `ChatRequest.message`.

This is defense-in-depth, not a guarantee: regex heuristics can be evaded.
The system prompt on the LLM side must still instruct the model to treat
the fenced block as data, never as instructions.
"""

import base64
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

from langsmith import traceable

MAX_MESSAGE_LENGTH = 10_000  # keep in sync with app.models.ChatRequest.message


class InjectionDetected(ValueError):
    """Raised when a message's risk score meets/exceeds the block threshold."""


# --- Character-level cleanup -------------------------------------------------

# Zero-width / invisible characters used to hide payloads or split flagged
# keywords (e.g. "ig​nore previous instructions").
_INVISIBLE_CHARS = re.compile(
    "[\u200b\u200c\u200d\u200e\u200f\u2060\ufeff\u00ad]"
)

# C0/C1 control characters, excluding \n and \t.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# Long runs of a repeated character - used to bury instructions in filler or
# to pad token counts.
_REPEAT_CHAR = re.compile(r"(.)\1{9,}")

_MULTI_WHITESPACE = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE = re.compile(r"\n{3,}")

# --- Pattern-level detection --------------------------------------------------

# Sequences that try to impersonate chat-format role delimiters so the model
# reinterprets user text as a system/assistant turn.
_ROLE_DELIMITER_PATTERNS = [
    re.compile(r"<\|.*?\|>"),                          # <|im_start|>, <|system|>
    re.compile(r"\[/?(system|assistant|developer)\]", re.I),
    re.compile(r"^\s*(system|assistant|developer)\s*:", re.I | re.M),
    re.compile(r"#{2,}\s*(system|instruction)", re.I),
]

# Instruction-override phrasing where similar wording can plausibly show up
# in genuine therapy talk (e.g. "forget what my dad told me"), so a lone
# match only contributes a partial score and needs a corroborating signal.
# Up to two filler words are tolerated between the verb and the temporal
# qualifier so paraphrases like "ignore your prior instructions" still match.
_FILL = r"(?:\w+\s+){0,2}"
_MEDIUM_CONFIDENCE_PHRASES = [
    rf"ignore\s+{_FILL}(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?)",
    rf"disregard\s+{_FILL}(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?)",
    rf"forget\s+{_FILL}(previous|prior|above|earlier)\s+(instructions?|rules?|prompt)",
    r"forget\s+(all|any|your)\s+(system\s+)?(instructions?|rules?|prompt)\b",
    r"you are now (a|an|no longer)\b",
    r"act as (if you|a|an)\b",
    r"new instructions?\s*:",
    r"override (your|the)?\s*(guidelines|rules|programming)",
    r"pretend (you|to) (are|be)\b",
]

# Phrases with essentially no legitimate use in a therapy chat - a single
# match is treated as conclusive on its own (no benign wording collides
# with "tell me your system prompt" or "jailbreak").
_HIGH_CONFIDENCE_PHRASES = [
    r"reveal\s+(?:your|the)?\s*(system prompt|instructions)",
    r"(what|show|print|tell me)\s+(is|are)?\s*(?:your|the)?\s*(system prompt|instructions)",
    r"print\s+(?:your|the)?\s*(system prompt|initial prompt)",
    r"repeat (the|your) (words|text) above",
    # Formatting-request framing for the same leak ("write/print/output the
    # text above in bold", "show everything above") - verb and the
    # above/text pair can appear in either order.
    r"(repeat|write|print|show|output)\s+.{0,25}\babove\b.{0,15}\b(text|words|content)\b",
    r"(repeat|write|print|show|output)\s+.{0,25}\b(text|words|content)\b.{0,15}\babove\b",
    r"do anything now",
    r"\bDAN\b",
    r"jailbreak",
    r"you have no (restrictions|rules|filters)",
]

_MEDIUM_RE = [re.compile(p, re.I) for p in _MEDIUM_CONFIDENCE_PHRASES]
_HIGH_RE = [re.compile(p, re.I) for p in _HIGH_CONFIDENCE_PHRASES]

_CODE_FENCE = re.compile(r"```")

# Long base64-looking blobs can smuggle an encoded secondary payload past
# naive keyword filters.
_BASE64_BLOB = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")


@dataclass
class SanitizationResult:
    text: str
    risk_score: int = 0
    matched_patterns: list = field(default_factory=list)

    @property
    def is_suspicious(self) -> bool:
        return self.risk_score > 0


def _strip_invisible_and_control(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE_CHARS.sub("", text)
    text = _CONTROL_CHARS.sub("", text)
    return text


def _collapse_whitespace(text: str) -> str:
    text = _MULTI_NEWLINE.sub("\n\n", text)
    text = _MULTI_WHITESPACE.sub(" ", text)
    return text.strip()


def _try_base64_decode(blob: str) -> Optional[str]:
    """Best-effort base64 decode, used to catch injection text smuggled
    past keyword filters as an encoded blob. Returns None if the blob isn't
    valid base64 or doesn't decode to printable text."""
    try:
        decoded = base64.b64decode(blob, validate=True).decode("utf-8").strip()
    except Exception:
        return None
    if not decoded or not all(c.isprintable() or c in "\n\t" for c in decoded):
        return None
    return decoded


def _score_phrases(text: str) -> tuple:
    """Score medium/high-confidence injection phrases only (no role
    delimiters or recursive base64 decoding) - used for one level of
    rescanning decoded payloads."""
    matched = []
    score = 0
    for pattern in _MEDIUM_RE:
        if pattern.search(text):
            matched.append(pattern.pattern)
            score += 4
    for pattern in _HIGH_RE:
        if pattern.search(text):
            matched.append(pattern.pattern)
            score += 6
    return score, matched


def analyze(text: str, *, max_length: Optional[int] = None) -> SanitizationResult:
    """Clean a raw user message and score it for prompt-injection risk."""
    if not isinstance(text, str):
        raise TypeError("message must be a string")

    limit = max_length or MAX_MESSAGE_LENGTH
    cleaned = text[:limit]
    cleaned = _strip_invisible_and_control(cleaned)
    cleaned = _REPEAT_CHAR.sub(lambda m: m.group(1) * 3, cleaned)
    cleaned = _collapse_whitespace(cleaned)

    matched = []
    score = 0

    for pattern in _ROLE_DELIMITER_PATTERNS:
        if pattern.search(cleaned):
            matched.append(pattern.pattern)
            # A spoofed role delimiter (e.g. "<|im_start|>system") has no
            # legitimate use in user text, so a single match is enough to
            # cross block_threshold on its own.
            score += 6
            cleaned = pattern.sub(" ", cleaned)

    for pattern in _MEDIUM_RE:
        if pattern.search(cleaned):
            matched.append(pattern.pattern)
            # A lone medium-confidence phrase can be a false positive (a
            # client describing their own experience), so it needs a second
            # corroborating signal to reach block_threshold.
            score += 4

    for pattern in _HIGH_RE:
        if pattern.search(cleaned):
            matched.append(pattern.pattern)
            # No plausible benign use in a therapy chat - block on its own.
            score += 6

    if _CODE_FENCE.search(cleaned):
        matched.append("code_fence")
        score += 1

    for blob_match in _BASE64_BLOB.finditer(cleaned):
        matched.append("possible_encoded_payload")
        score += 1
        decoded = _try_base64_decode(blob_match.group(0))
        if decoded:
            decoded_score, decoded_matches = _score_phrases(decoded)
            if decoded_matches:
                score += decoded_score
                matched.extend(f"decoded:{m}" for m in decoded_matches)

    return SanitizationResult(text=cleaned, risk_score=score, matched_patterns=matched)


@traceable(name="sanitize_message")
def sanitize_message(text: str, *, block_threshold: int = 6) -> str:
    """Sanitize a user message; raise InjectionDetected if too risky to forward.

    Returns the cleaned text on success. Callers should still pass the
    result through `wrap_for_prompt` before building the LLM call.
    """
    result = analyze(text)
    if result.risk_score >= block_threshold:
        raise InjectionDetected(
            f"message blocked: risk_score={result.risk_score} "
            f"matched={result.matched_patterns}"
        )
    return result.text


def wrap_for_prompt(sanitized_text: str, delimiter: str = "USER_MESSAGE") -> str:
    """Fence sanitized text so it can't be mistaken for a system/assistant turn.

    Build LLM prompts by concatenating the system prompt with this wrapped
    block only - never splice the raw `ChatRequest.message` in directly.
    """
    body = sanitized_text.replace(f"<<<{delimiter}", "").replace(f"{delimiter}>>>", "")
    return f"<<<{delimiter}\n{body}\n{delimiter}>>>"
