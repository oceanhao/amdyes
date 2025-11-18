# my_rewards.py
#TODO:增加反事实
import re

def _normalize(s: str) -> str:
    return re.sub(r"\s+", "", s.strip())

def _format_ok(s: str) -> bool:
    return "####" in s  # 按需改

def compute_score(*, data_source: str, solution_str: str, ground_truth: dict | str,
                  extra_info: dict | None = None,
                  alpha: float = 1.0, beta: float = 0.2, lambda_coef: float = 0.0) -> float:
    acc = 1.0 if _normalize(solution_str) == _normalize(ground_truth) else 0.0
    fmt = 1.0 if _format_ok(solution_str) else 0.0
    score = alpha*acc + beta*fmt
    return {"score": float(score), "acc": float(acc), "format": float(fmt)}
