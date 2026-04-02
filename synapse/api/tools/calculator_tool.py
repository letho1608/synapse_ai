"""Máy tính - tính biểu thức toán an toàn. Không cần API."""
import ast
import operator
import math
from typing import Any, Dict

# Cho phép các phép toán an toàn
BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}
SAFE_NAMES = {"abs": abs, "round": round, "min": min, "max": max, "sum": sum}
SAFE_NAMES.update({k: v for k, v in math.__dict__.items() if not k.startswith("_")})


def _eval_node(node: ast.AST):
    if isinstance(node, ast.Constant):
        return node.value
    if hasattr(ast, "Num") and isinstance(node, ast.Num):
        return node.n
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp):
        return UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in SAFE_NAMES:
            args = [_eval_node(a) for a in node.args]
            return SAFE_NAMES[node.func.id](*args)
    raise ValueError("Biểu thức không được phép")


async def calculate(expression: str) -> Dict[str, Any]:
    """
    Tính biểu thức toán học an toàn.
    Ví dụ: "2 + 3 * 4", "sin(0.5)", "sqrt(16)"
    """
    try:
        expr = expression.strip().replace(",", ".")
        tree = ast.parse(expr, mode="eval")
        result = _eval_node(tree.body)
        if isinstance(result, float) and result == int(result):
            result = int(result)
        return {"success": True, "expression": expression, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e), "expression": expression}
