#!/usr/bin/env python3
"""Generate MDX API reference docs from Unforget source code.

Parses src/unforget/*.py with the ast module, extracts public classes,
methods, functions, and their signatures/docstrings, and writes MDX files
ready for the Nextra docs site.

Usage:
    python scripts/generate_api_docs.py              # writes to docs_out/
    python scripts/generate_api_docs.py --out ../docs/app/docs/api-reference/generated
"""

from __future__ import annotations

import ast
import argparse
import textwrap
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src" / "unforget"

# Modules to document and their display order
MODULES = [
    ("store", "MemoryStore", "Core store class — all memory operations go through here."),
    ("scoped", "ScopedMemory", "Pre-bound store with org_id/agent_id for cleaner usage."),
    ("types", "Types & Models", "Pydantic models, enums, and type definitions."),
    ("api", "FastAPI Router", "REST API router with 17 endpoints."),
    ("tools", "Memory Tools", "LLM-callable tools and executor."),
    ("integrations/openai", "OpenAI Integration", "Wrap AsyncOpenAI with automatic memory."),
    ("integrations/anthropic", "Anthropic Integration", "Wrap AsyncAnthropic with automatic memory."),
    ("embedder", "Embedders", "Embedding model interfaces and implementations."),
    ("retrieval", "Retrieval", "4-channel hybrid retrieval with RRF fusion."),
    ("consolidation", "Consolidation", "Background dedup, decay, and promotion."),
    ("ingest", "Ingestion", "Conversation ingestion in multiple modes."),
    ("temporal", "Temporal", "Versioning, supersession, and timeline queries."),
    ("scheduler", "Scheduler", "Background consolidation scheduler."),
    ("entities", "Entities", "Named entity extraction."),
    ("cache", "Cache", "Embedding and recall caching."),
    ("quotas", "Quotas", "Rate limiting and memory quotas."),
    ("schema", "Schema", "Database schema management."),
]


def get_source(module_path: str) -> str:
    """Read source file content."""
    path = SRC_DIR / f"{module_path}.py"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def extract_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Extract a function/method signature as a string."""
    args = node.args
    parts = []

    # Regular args
    num_args = len(args.args)
    num_defaults = len(args.defaults)
    non_default_count = num_args - num_defaults

    for i, arg in enumerate(args.args):
        if arg.arg == "self" or arg.arg == "cls":
            continue
        name = arg.arg
        annotation = ""
        if arg.annotation:
            annotation = f": {ast.unparse(arg.annotation)}"

        if i >= non_default_count:
            default_idx = i - non_default_count
            default = ast.unparse(args.defaults[default_idx])
            parts.append(f"{name}{annotation} = {default}")
        else:
            parts.append(f"{name}{annotation}")

    # *args
    if args.vararg:
        name = args.vararg.arg
        ann = f": {ast.unparse(args.vararg.annotation)}" if args.vararg.annotation else ""
        parts.append(f"*{name}{ann}")
    elif args.kwonlyargs:
        parts.append("*")

    # keyword-only args
    for i, kwarg in enumerate(args.kwonlyargs):
        name = kwarg.arg
        annotation = f": {ast.unparse(kwarg.annotation)}" if kwarg.annotation else ""
        if i < len(args.kw_defaults) and args.kw_defaults[i] is not None:
            default = ast.unparse(args.kw_defaults[i])
            parts.append(f"{name}{annotation} = {default}")
        else:
            parts.append(f"{name}{annotation}")

    # **kwargs
    if args.kwarg:
        name = args.kwarg.arg
        ann = f": {ast.unparse(args.kwarg.annotation)}" if args.kwarg.annotation else ""
        parts.append(f"**{name}{ann}")

    sig = ", ".join(parts)

    # Return annotation
    ret = ""
    if node.returns:
        ret = f" -> {ast.unparse(node.returns)}"

    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return f"{prefix} {node.name}({sig}){ret}"


def extract_docstring(node: ast.AST) -> str:
    """Extract docstring from a node."""
    if (
        node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    ):
        return textwrap.dedent(node.body[0].value.value).strip()
    return ""


def extract_class_info(node: ast.ClassDef) -> dict:
    """Extract class name, docstring, and public methods."""
    methods = []
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if item.name.startswith("_") and item.name != "__init__":
                continue
            methods.append({
                "name": item.name,
                "signature": extract_signature(item),
                "docstring": extract_docstring(item),
                "is_async": isinstance(item, ast.AsyncFunctionDef),
                "is_property": any(
                    isinstance(d, ast.Name) and d.id == "property"
                    for d in item.decorator_list
                ),
            })

    return {
        "name": node.name,
        "docstring": extract_docstring(node),
        "methods": methods,
        "bases": [ast.unparse(b) for b in node.bases],
    }


def extract_module_info(source: str) -> dict:
    """Extract all public classes, functions, and constants from a module."""
    tree = ast.parse(source)
    module_doc = extract_docstring(tree)

    classes = []
    functions = []
    constants = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            classes.append(extract_class_info(node))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            functions.append({
                "name": node.name,
                "signature": extract_signature(node),
                "docstring": extract_docstring(node),
                "is_async": isinstance(node, ast.AsyncFunctionDef),
            })
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    try:
                        value = ast.unparse(node.value)
                        if len(value) < 200:
                            constants.append({"name": target.id, "value": value})
                    except Exception:
                        pass

    return {
        "module_doc": module_doc,
        "classes": classes,
        "functions": functions,
        "constants": constants,
    }


def format_method_mdx(method: dict, class_name: str = "") -> str:
    """Format a single method as MDX."""
    lines = []

    if method.get("is_property"):
        lines.append(f"#### `{method['name']}` (property)")
    else:
        display_name = method["name"]
        if display_name == "__init__":
            display_name = f"{class_name}()" if class_name else "__init__"
        lines.append(f"#### `{display_name}`")

    lines.append("")
    lines.append("```python")
    lines.append(method["signature"])
    lines.append("```")

    if method["docstring"]:
        lines.append("")
        # Take first paragraph of docstring
        first_para = method["docstring"].split("\n\n")[0]
        lines.append(first_para)

    lines.append("")
    return "\n".join(lines)


def generate_module_mdx(module_path: str, title: str, description: str) -> str:
    """Generate MDX content for a module."""
    source = get_source(module_path)
    if not source:
        return ""

    info = extract_module_info(source)
    lines = [
        "{/* Auto-generated from source — do not edit manually */}",
        "",
        f"# {title}",
        "",
        description,
        "",
        f"`src/unforget/{module_path}.py`",
        "",
    ]

    # Classes
    for cls in info["classes"]:
        lines.append(f"## `{cls['name']}`")
        lines.append("")
        if cls["bases"]:
            lines.append(f"Extends: `{'`, `'.join(cls['bases'])}`")
            lines.append("")
        if cls["docstring"]:
            first_para = cls["docstring"].split("\n\n")[0]
            lines.append(first_para)
            lines.append("")

        # Constructor first
        init_methods = [m for m in cls["methods"] if m["name"] == "__init__"]
        other_methods = [m for m in cls["methods"] if m["name"] != "__init__"]
        properties = [m for m in other_methods if m.get("is_property")]
        regular = [m for m in other_methods if not m.get("is_property")]

        if init_methods:
            lines.append("### Constructor")
            lines.append("")
            lines.append(format_method_mdx(init_methods[0], cls["name"]))

        if properties:
            lines.append("### Properties")
            lines.append("")
            for prop in properties:
                lines.append(format_method_mdx(prop, cls["name"]))

        if regular:
            lines.append("### Methods")
            lines.append("")
            for method in regular:
                lines.append(format_method_mdx(method, cls["name"]))

        lines.append("---")
        lines.append("")

    # Module-level functions
    if info["functions"]:
        lines.append("## Functions")
        lines.append("")
        for func in info["functions"]:
            lines.append(f"### `{func['name']}`")
            lines.append("")
            lines.append("```python")
            lines.append(func["signature"])
            lines.append("```")
            if func["docstring"]:
                lines.append("")
                first_para = func["docstring"].split("\n\n")[0]
                lines.append(first_para)
            lines.append("")

    # Constants
    if info["constants"]:
        lines.append("## Constants")
        lines.append("")
        for const in info["constants"]:
            lines.append(f"- `{const['name']}` = `{const['value']}`")
        lines.append("")

    return "\n".join(lines)


def generate_index_mdx(modules: list[tuple[str, str, str]]) -> str:
    """Generate the index page for the API reference."""
    lines = [
        "{/* Auto-generated from source — do not edit manually */}",
        "",
        "# API Reference",
        "",
        "Auto-generated from the Unforget source code.",
        "",
        "## Modules",
        "",
        "| Module | Description |",
        "|--------|-------------|",
    ]
    for module_path, title, description in modules:
        slug = module_path.replace("/", "-")
        lines.append(f"| [`{title}`](/docs/api-reference/generated/{slug}) | {description} |")

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate MDX API docs from source")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent / "docs_out"),
        help="Output directory for MDX files",
    )
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating API docs from {SRC_DIR}")
    print(f"Output: {out_dir}")
    print()

    generated = 0
    for module_path, title, description in MODULES:
        mdx = generate_module_mdx(module_path, title, description)
        if not mdx:
            print(f"  SKIP {module_path} (not found)")
            continue

        slug = module_path.replace("/", "-")
        file_path = out_dir / slug
        file_path.mkdir(parents=True, exist_ok=True)
        (file_path / "page.mdx").write_text(mdx, encoding="utf-8")
        generated += 1
        print(f"  OK   {module_path} → {slug}/page.mdx")

    # Generate index
    index_mdx = generate_index_mdx(MODULES)
    (out_dir / "page.mdx").write_text(index_mdx, encoding="utf-8")

    print(f"\nGenerated {generated} module docs + index")


if __name__ == "__main__":
    main()
