"""
结构约束测试。

把架构边界写成可执行约束，避免 API / 前端依赖倒灌进核心模块。
"""

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LAYER_RULES = {
    "src/retrieval": ("src.api", "src.frontend"),
    "src/llm": ("src.api", "src.frontend"),
    "src/knowledge_graph": ("src.api", "src.frontend"),
    "src/data_processing": ("src.api", "src.frontend"),
}


def _iter_imports(file_path: Path) -> list[str]:
    tree = ast.parse(file_path.read_text(encoding="utf-8"))
    imports: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imports.append(node.module)

    return imports


def _is_forbidden(module_name: str, forbidden_prefixes: tuple[str, ...]) -> bool:
    return any(
        module_name == prefix or module_name.startswith(f"{prefix}.")
        for prefix in forbidden_prefixes
    )


def test_core_layers_do_not_depend_on_api_or_frontend():
    violations: list[str] = []

    for layer, forbidden_prefixes in LAYER_RULES.items():
        for file_path in (PROJECT_ROOT / layer).rglob("*.py"):
            imports = _iter_imports(file_path)
            offenders = sorted(
                {module_name for module_name in imports if _is_forbidden(module_name, forbidden_prefixes)}
            )
            if offenders:
                violations.append(f"{file_path.relative_to(PROJECT_ROOT)} -> {', '.join(offenders)}")

    assert not violations, "发现越层依赖:\n" + "\n".join(violations)
