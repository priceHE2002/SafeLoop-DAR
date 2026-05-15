from __future__ import annotations

import re


_NUMBER = re.compile(r"^-?\d+(\.\d+)?$")
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def simple_tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def classify_token(token: str, stage: str = "unknown") -> str:
    if not token:
        return "unknown"
    if token in {".", ",", ":", ";", "!", "?", "(", ")", "[", "]", "{", "}"}:
        return "punctuation"
    if _NUMBER.match(token):
        if stage == "final_answer":
            return "math_final_number"
        return "number"
    if stage == "tool_call_json":
        if token in {"tool", "arguments", "name", "expression"}:
            return "json_key"
        return "json_value"
    if stage == "code_generation" and _IDENT.match(token):
        return "code_identifier"
    if stage in {"retrieval", "final_answer"} and token[:1].isupper():
        return "retrieved_entity"
    if token.lower() in {"the", "a", "an", "is", "are", "to", "of", "and"}:
        return "normal_text"
    return "normal_text"

