import contextlib
import io
import json
import sys
import traceback
from pathlib import Path


def execute_notebook(notebook_path: Path) -> None:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    globals_dict = {"__name__": "__main__"}
    execution_count = 1

    for index, cell in enumerate(notebook.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue

        source = "".join(cell.get("source", []))
        stdout = io.StringIO()
        stderr = io.StringIO()
        cell["execution_count"] = execution_count
        cell["outputs"] = []

        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(compile(source, f"{notebook_path.name}:cell_{index}", "exec"), globals_dict)
        except Exception:
            traceback.print_exc(file=stderr)
            cell["outputs"].append(
                {
                    "name": "stderr",
                    "output_type": "stream",
                    "text": stderr.getvalue(),
                }
            )
            notebook_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
            raise

        if stdout.getvalue():
            cell["outputs"].append(
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": stdout.getvalue(),
                }
            )
        if stderr.getvalue():
            cell["outputs"].append(
                {
                    "name": "stderr",
                    "output_type": "stream",
                    "text": stderr.getvalue(),
                }
            )
        execution_count += 1

    notebook_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: execute_notebook.py <notebook-path>")
    execute_notebook(Path(sys.argv[1]).resolve())
