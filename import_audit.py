#!/usr/bin/env python3
from __future__ import annotations

"""
Import audit that is repo-structure aware and does NOT flag stdlib/external packages as missing.

It reports:
A) INTERNAL IMPORTS THAT WILL BREAK:
   - internal import that does not resolve under root nor src/
   - internal import that resolves only under src/ but is imported without `src.` (src-layout mismatch)
   - internal import that is ambiguous (exists in both root and src)

B) PACKAGE STRUCTURE WARNINGS:
   - internal import that requires directories to be packages but __init__.py is missing
     (you can choose to treat this as warning or error)

C) OPTIONAL INVENTORY:
   - list external imports (third-party deps) used in the repo (not errors)

This is fully offline: no importing, no env introspection.
"""

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

EXCLUDE_DIRS = {
    ".git", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".venv", "venv", "build", "dist", "node_modules",
}

# Python stdlib module list (offline, reliable) for your current interpreter
try:
    STDLIB_MODULES: Set[str] = set(sys.stdlib_module_names)  # py>=3.10
except Exception:
    STDLIB_MODULES = set()

@dataclass(frozen=True)
class ImportRef:
    file: Path
    lineno: int
    col: int
    kind: str      # "import" | "from"
    module: str    # e.g. "utils.notreks" or "numpy"
    raw: str       # full statement (pretty)

@dataclass(frozen=True)
class ExistsResult:
    ok: bool
    reason: str
    missing_init_paths: Tuple[str, ...]  # directories that should have __init__.py but don't

@dataclass(frozen=True)
class Resolution:
    root: ExistsResult
    src: ExistsResult

def iter_py_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.py"):
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        yield p

def is_package_dir(d: Path) -> bool:
    return d.is_dir() and (d / "__init__.py").exists()

def discover_top_level_modules(root: Path) -> Set[str]:
    names: Set[str] = set()
    if not root.exists():
        return names
    for child in root.iterdir():
        if child.name in EXCLUDE_DIRS:
            continue
        if child.is_file() and child.suffix == ".py":
            names.add(child.stem)
        elif is_package_dir(child):
            names.add(child.name)
        # NOTE: we intentionally do NOT treat directories without __init__.py as packages here,
        # because that is exactly the brittle thing we want to warn about.
    return names

def module_exists_under(base: Path, dotted: str) -> ExistsResult:
    """
    Check if dotted module exists under base as package chain or module file.
    Additionally record missing __init__.py along the chain.
    """
    parts = dotted.split(".")
    cur = base
    missing_inits: List[str] = []

    for i, part in enumerate(parts):
        pkg_dir = cur / part
        mod_file = cur / f"{part}.py"

        last = (i == len(parts) - 1)

        if last:
            if pkg_dir.is_dir():
                if not (pkg_dir / "__init__.py").exists():
                    missing_inits.append(str(pkg_dir.relative_to(base)))
                # Even without __init__.py, it might work as namespace package,
                # but that’s environment-dependent. We treat as "ok but warn".
                return ExistsResult(True, f"dir {pkg_dir.relative_to(base)}", tuple(missing_inits))
            if mod_file.exists():
                return ExistsResult(True, f"file {mod_file.relative_to(base)}", tuple(missing_inits))
            return ExistsResult(False, f"not found: {pkg_dir.relative_to(base)} or {mod_file.relative_to(base)}", tuple(missing_inits))

        # intermediate segments must be package dirs
        if pkg_dir.is_dir():
            if not (pkg_dir / "__init__.py").exists():
                missing_inits.append(str(pkg_dir.relative_to(base)))
            cur = pkg_dir
            continue

        # maybe there is a module file at an intermediate segment: then dotted import can't proceed
        if mod_file.exists():
            return ExistsResult(False, f"blocked by file {mod_file.relative_to(base)} (not a package dir)", tuple(missing_inits))

        return ExistsResult(False, f"missing dir {pkg_dir.relative_to(base)}", tuple(missing_inits))

    return ExistsResult(False, "unreachable", tuple(missing_inits))

def parse_imports(path: Path) -> List[ImportRef]:
    txt = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(txt, filename=str(path))
    except SyntaxError:
        return []

    out: List[ImportRef] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name
                out.append(ImportRef(path, node.lineno, node.col_offset, "import", mod, f"import {mod}"))
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue
            if node.module is None:
                continue
            mod = node.module
            imported = ", ".join(a.name for a in node.names)
            out.append(ImportRef(path, node.lineno, node.col_offset, "from", mod, f"from {mod} import {imported}"))
    return out

def top_level_name(mod: str) -> str:
    return mod.split(".", 1)[0]

def is_stdlib(name: str) -> bool:
    # __future__ is special-cased: appears in sys.stdlib_module_names but still fine
    if name == "__future__":
        return True
    return name in STDLIB_MODULES

def resolve(mod: str, root: Path, src: Path) -> Resolution:
    return Resolution(
        root=module_exists_under(root, mod),
        src=module_exists_under(src, mod) if src.exists() else ExistsResult(False, "src/ missing", ()),
    )

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--src-dir", type=str, default="src")
    ap.add_argument("--max", type=int, default=300)
    ap.add_argument("--show-external", action="store_true", help="Also print external deps inventory.")
    ap.add_argument("--treat-missing-init-as-error", action="store_true",
                    help="If set, missing __init__.py along internal imports is an error instead of warning.")
    args = ap.parse_args()

    root = args.root.resolve()
    src = (root / args.src_dir).resolve()

    top_root = discover_top_level_modules(root)
    top_src = discover_top_level_modules(src) if src.exists() else set()
    internal_toplevel = top_root | top_src

    print(f"[import_audit] root={root}")
    print(f"[import_audit] src ={src} {'(exists)' if src.exists() else '(missing)'}")
    print(f"[import_audit] internal top-level names: {sorted(internal_toplevel)}")
    print()

    refs: List[ImportRef] = []
    for f in iter_py_files(root):
        refs.extend(parse_imports(f))

    internal_refs: List[Tuple[ImportRef, Resolution]] = []
    external_refs: List[ImportRef] = []
    stdlib_refs: List[ImportRef] = []

    for r in refs:
        tl = top_level_name(r.module)
        if tl in internal_toplevel:
            internal_refs.append((r, resolve(r.module, root, src)))
        elif is_stdlib(tl):
            stdlib_refs.append(r)
        else:
            external_refs.append(r)

    # classify internal issues
    src_only: List[Tuple[ImportRef, Resolution]] = []
    missing: List[Tuple[ImportRef, Resolution]] = []
    ambiguous: List[Tuple[ImportRef, Resolution]] = []
    missing_init: List[Tuple[ImportRef, Resolution, str]] = []
    ok: List[Tuple[ImportRef, Resolution]] = []

    for r, res in internal_refs:
        can_root = res.root.ok
        can_src = res.src.ok

        if not can_root and not can_src:
            missing.append((r, res))
            continue

        if can_root and can_src:
            ambiguous.append((r, res))
        elif (not can_root) and can_src:
            # exists only under src layout, imported without src.<...> typically
            src_only.append((r, res))
        else:
            ok.append((r, res))

        # collect missing __init__.py warnings (namespace packages are brittle on clusters)
        miss = set(res.root.missing_init_paths) | set(res.src.missing_init_paths)
        for m in sorted(miss):
            missing_init.append((r, res, m))

    # pretty printer
    printed = 0
    def section(title: str):
        print(title)
        print("-" * len(title))

    def show(group: List, title: str, render):
        nonlocal printed
        if not group:
            return
        section(title)
        for item in group[: args.max]:
            render(item)
            printed += 1
            if printed >= args.max:
                print(f"... reached --max={args.max}, stopping output.")
                break
        print()

    def render_internal(item: Tuple[ImportRef, Resolution]):
        r, res = item
        rel = r.file.relative_to(root)
        print(f"{rel}:{r.lineno}:{r.col}")
        print(f"  {r.raw}")
        print(f"  root: {res.root.ok} ({res.root.reason})")
        print(f"  src : {res.src.ok} ({res.src.reason})")

    def render_missing_init(item: Tuple[ImportRef, Resolution, str]):
        r, res, m = item
        rel = r.file.relative_to(root)
        print(f"{rel}:{r.lineno}:{r.col}")
        print(f"  {r.raw}")
        print(f"  ⚠ missing __init__.py in: {m}")

    show(missing, "INTERNAL IMPORT ERROR: not found in repo-root nor src/", render_internal)
    show(src_only, "INTERNAL IMPORT LIKELY TO BREAK ON CLUSTER: exists under src/ but imported without `src.`", render_internal)
    show(ambiguous, "INTERNAL IMPORT ENV-DEPENDENT: resolves in both root and src (PYTHONPATH-sensitive)", render_internal)

    # missing init paths: either warning or error based on flag
    if missing_init:
        title = "PACKAGE STRUCTURE " + ("ERROR" if args.treat_missing_init_as_error else "WARNING") + ": missing __init__.py on import path"
        show(missing_init, title, render_missing_init)

    if args.show_external:
        # inventory unique external top-level names
        deps = sorted({top_level_name(r.module) for r in external_refs})
        section("EXTERNAL DEPENDENCIES (inventory, not errors)")
        print(", ".join(deps))
        print()

    # summary
    section("SUMMARY")
    print(f"Python files scanned: {len(list(iter_py_files(root)))}")
    print(f"Absolute imports scanned: {len(refs)}")
    print(f"Internal imports scanned: {len(internal_refs)}")
    print(f"  missing:     {len(missing)}")
    print(f"  src-only:    {len(src_only)}")
    print(f"  ambiguous:   {len(ambiguous)}")
    print(f"  ok:          {len(ok)}")
    print(f"  missing_init warnings: {len(missing_init)}")
    print(f"Stdlib imports ignored:   {len(stdlib_refs)}")
    print(f"External imports ignored: {len(external_refs)}")

    has_error = bool(missing or src_only or ambiguous or (args.treat_missing_init_as_error and missing_init))
    return 1 if has_error else 0

if __name__ == "__main__":
    raise SystemExit(main())

    
    
# !python import_audit.py --root . --show-ok
