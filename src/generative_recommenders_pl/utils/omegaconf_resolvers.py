"""OmegaConf resolver registration helpers."""

from __future__ import annotations

import ast
import operator
from typing import Any

from omegaconf import OmegaConf


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_arithmetic_eval(expr: Any) -> int | float:
    """Evaluate numeric config expressions without exposing Python eval."""

    if isinstance(expr, (int, float)):
        return expr
    node = ast.parse(str(expr), mode="eval")

    def visit(current: ast.AST) -> int | float:
        if isinstance(current, ast.Expression):
            return visit(current.body)
        if isinstance(current, ast.Constant) and isinstance(current.value, (int, float)):
            return current.value
        if isinstance(current, ast.BinOp) and type(current.op) in _BIN_OPS:
            return _BIN_OPS[type(current.op)](visit(current.left), visit(current.right))
        if isinstance(current, ast.UnaryOp) and type(current.op) in _UNARY_OPS:
            return _UNARY_OPS[type(current.op)](visit(current.operand))
        raise ValueError(f"Unsupported resolver expression: {expr!r}")

    value = visit(node)
    return int(value) if isinstance(value, float) and value.is_integer() else value


def register_safe_resolvers() -> None:
    OmegaConf.register_new_resolver("eval", safe_arithmetic_eval, replace=True)
