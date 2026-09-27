from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from dataset_manager import Dataset


ALLOWED_IMPORTS = {"pandas", "numpy", "matplotlib", "seaborn", "plotly", "sklearn"}
FORBIDDEN_CALLS = {"open", "eval", "exec", "compile", "__import__", "input", "breakpoint"}
FORBIDDEN_ATTRIBUTES = {
    "system", "popen", "spawn", "fork", "remove", "unlink", "rmdir", "rename",
    "replace", "chmod", "chown", "walk", "listdir", "scandir", "read_pickle",
    "read_csv", "read_excel", "read_json", "read_parquet", "read_pickle", "read_sql",
    "read_html", "read_xml", "read_clipboard", "to_sql", "to_pickle", "to_csv",
    "to_excel", "to_json", "to_parquet", "savefig",
}


class UnsafeCodeError(ValueError):
    pass


@dataclass
class ExecutionResult:
    success: bool
    summary: object | None = None
    tables: list[dict] = field(default_factory=list)
    chart_paths: list[str] = field(default_factory=list)
    stdout: str = ""
    error: str | None = None

    def compact_for_agent(self) -> dict:
        return {
            "success": self.success,
            "summary": self.summary,
            "tables": [
                {"name": table["name"], "columns": table["columns"], "rows": table["rows"][:30]}
                for table in self.tables
            ],
            "charts_created": [Path(path).name for path in self.chart_paths],
            "error": self.error,
        }


def validate_generated_code(code: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise UnsafeCodeError(f"Generated code has invalid Python syntax: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [alias.name.split(".")[0] for alias in node.names] if isinstance(node, ast.Import) else [(node.module or "").split(".")[0]]
            denied = [name for name in modules if name not in ALLOWED_IMPORTS]
            if denied:
                raise UnsafeCodeError(f"Import not allowed: {', '.join(denied)}")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr in FORBIDDEN_ATTRIBUTES:
                raise UnsafeCodeError(f"Operation not allowed: {node.attr}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            raise UnsafeCodeError(f"Function not allowed: {node.func.id}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise UnsafeCodeError("Dunder names are not allowed.")


def _wrapper(code: str) -> str:
    return f'''import json
import math
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

BASE_DIR = Path.cwd()
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
with (BASE_DIR / "manifest.json").open("r", encoding="utf-8") as handle:
    manifest = json.load(handle)
datasets = {{name: pd.read_parquet(BASE_DIR / path) for name, path in manifest.items()}}
chart_paths = []

def save_chart(figure, name):
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name)).strip("_")[:60] or "chart"
    path = OUTPUT_DIR / f"{{len(chart_paths) + 1:02d}}_{{safe_name}}.png"
    figure.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    chart_paths.append(path.name)

def save_plotly(figure, name):
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name)).strip("_")[:60] or "chart"
    path = OUTPUT_DIR / f"{{len(chart_paths) + 1:02d}}_{{safe_name}}.plotly.json"
    path.write_text(figure.to_json(), encoding="utf-8")
    chart_paths.append(path.name)

result = {{}}
tables = {{}}

# ---- agent-generated analysis ----
{code}
# ---- end generated analysis ----

def clean(value):
    if isinstance(value, dict):
        return {{str(k): clean(v) for k, v in value.items()}}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_,)): return bool(value)
    if isinstance(value, (pd.Timestamp,)): return value.isoformat()
    if pd.isna(value): return None
    return value

table_payload = []
if not isinstance(tables, dict):
    raise TypeError("tables must be a dictionary of names to DataFrames")
for name, frame in tables.items():
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Table '{{name}}' is not a pandas DataFrame")
    preview = frame.head(100)
    table_payload.append({{"name": str(name), "columns": [str(c) for c in preview.columns], "rows": clean(preview.to_dict(orient="records"))}})

payload = {{"summary": clean(result), "tables": table_payload, "charts": chart_paths}}
with (OUTPUT_DIR / "result.json").open("w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False)
'''


def execute_analysis(code: str, datasets: list[Dataset], timeout_seconds: int = 25) -> ExecutionResult:
    try:
        validate_generated_code(code)
    except UnsafeCodeError as exc:
        return ExecutionResult(False, error=str(exc))

    with tempfile.TemporaryDirectory(prefix="analyst_exec_") as temp:
        working = Path(temp)
        manifest = {}
        for dataset in datasets:
            filename = f"{dataset.dataset_id}.parquet"
            dataset.frame.to_parquet(working / filename, index=False)
            manifest[dataset.dataset_id] = filename
        (working / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        script = working / "analysis.py"
        script.write_text(_wrapper(code), encoding="utf-8")

        # Reusing the font cache keeps each isolated chart process fast.
        matplotlib_cache = Path(tempfile.gettempdir()) / "northstar_mpl_cache"
        matplotlib_cache.mkdir(exist_ok=True)
        env = {"PATH": os.environ.get("PATH", ""), "MPLCONFIGDIR": str(matplotlib_cache)}
        try:
            process = subprocess.run(
                [sys.executable, "-I", str(script)],
                cwd=working,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(False, error=f"Analysis exceeded the {timeout_seconds}-second time limit.")
        except Exception as exc:
            return ExecutionResult(False, error=f"The analysis process could not start: {exc}")

        if process.returncode != 0:
            error = (process.stderr or process.stdout or "Unknown execution error")[-4000:]
            error = error.replace(str(working), "<workspace>")
            return ExecutionResult(False, stdout=process.stdout[-1000:], error=error)

        result_file = working / "output" / "result.json"
        if not result_file.exists():
            return ExecutionResult(False, error="The code completed but did not produce a result artifact.")
        try:
            payload = json.loads(result_file.read_text(encoding="utf-8"))
        except Exception as exc:
            return ExecutionResult(False, error=f"The generated result could not be read: {exc}")

        # Copy chart bytes into memory-independent temporary files retained by Streamlit.
        retained_dir = Path(tempfile.mkdtemp(prefix="analyst_charts_"))
        chart_paths = []
        for filename in payload.get("charts", []):
            source = working / "output" / Path(filename).name
            if source.exists():
                target = retained_dir / source.name
                target.write_bytes(source.read_bytes())
                chart_paths.append(str(target))
        return ExecutionResult(True, payload.get("summary"), payload.get("tables", []), chart_paths, process.stdout[-1000:])
