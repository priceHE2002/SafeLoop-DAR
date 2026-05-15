from __future__ import annotations


HIGH_RISK_TYPES = {
    "math_final_number",
    "number",
    "code_identifier",
    "json_value",
    "tool_argument",
    "retrieved_entity",
    "citation_entity",
}


def risk_group(token_type: str, stage: str) -> str:
    if stage == "tool_call_json":
        return "tool_json"
    if stage == "code_generation":
        return "code"
    if stage == "final_answer" and token_type in {"math_final_number", "number"}:
        return "math_final"
    if token_type in {"retrieved_entity", "citation_entity"}:
        return "entity"
    if token_type in {"punctuation", "normal_text"}:
        return "low_risk_text"
    return "general"


def token_stage_features(token_type: str, stage: str) -> dict[str, float]:
    group = risk_group(token_type, stage)
    return {
        "is_high_risk_token": 1.0 if token_type in HIGH_RISK_TYPES else 0.0,
        "is_tool_stage": 1.0 if stage == "tool_call_json" else 0.0,
        "is_code_stage": 1.0 if stage == "code_generation" else 0.0,
        "is_final_stage": 1.0 if stage == "final_answer" else 0.0,
        "group_low_risk_text": 1.0 if group == "low_risk_text" else 0.0,
        "group_tool_json": 1.0 if group == "tool_json" else 0.0,
        "group_code": 1.0 if group == "code" else 0.0,
        "group_math_final": 1.0 if group == "math_final" else 0.0,
        "group_entity": 1.0 if group == "entity" else 0.0,
    }

