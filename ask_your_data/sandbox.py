"""run_python: pandas analysis in a restricted subprocess.

NOT a hardened sandbox. It is a best-effort guard for model-written analysis code:

* separate interpreter process in isolated mode (``-I``), reduced environment,
  working directory = a throwaway temp dir
* wall-clock timeout (process is killed) and output truncation
* a Python audit hook that blocks network, subprocesses and file writes outside the temp dir
* the code never sees a database connection: it receives the result of a guarded SELECT as ``df``

Audit hooks can be bypassed by determined code (e.g. native extensions), and there is no
memory/CPU cap on Windows. Do not run untrusted users' code with this; use a container/VM.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

MAX_OUTPUT_CHARS = 6000

_WRAPPER = r"""
import os, sys
tmp = os.path.realpath(sys.argv[1])
import numpy as np
import pandas as pd

df = pd.read_json(os.path.join(tmp, "input.json"), orient="table")
code = open(os.path.join(tmp, "user_code.py"), encoding="utf-8").read()

_BLOCKED = (
    "socket.", "subprocess.Popen", "os.system", "os.exec", "os.spawn", "os.posix_spawn",
    "os.remove", "os.rename", "os.rmdir", "os.mkdir", "os.chmod", "os.symlink", "os.link",
    "shutil.", "ctypes.", "os.startfile", "webbrowser.",
)

def _inside(path):
    try:
        return os.path.realpath(os.fsdecode(path)).startswith(tmp)
    except Exception:
        return False

def _hook(event, args):
    if event.startswith(_BLOCKED):
        raise PermissionError("blocked by sandbox: " + event)
    if event == "open":
        path, mode, flags = args
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
            isinstance(flags, int)
            and flags & (os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC))
        if writing and not _inside(path):
            raise PermissionError("blocked by sandbox: writing outside the temp dir")

sys.addaudithook(_hook)
exec(compile(code, "<analysis>", "exec"), {"df": df, "pd": pd, "np": np, "__name__": "__main__"})
"""


@dataclass
class PythonResult:
    ok: bool
    stdout: str
    stderr: str
    timed_out: bool = False

    def to_text(self) -> str:
        if self.timed_out:
            return "ERROR: python timed out.\n" + self.stdout
        if not self.ok:
            tail = self.stderr[-1500:]
            extra = f"\nstdout so far:\n{self.stdout}" if self.stdout else ""
            return "ERROR:\n" + tail + extra
        return self.stdout or "(no output - use print() to show results)"


def _clip(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + f"\n... [output truncated at {MAX_OUTPUT_CHARS} chars]"


def run_python(code: str, frame: pd.DataFrame, timeout_s: float = 10.0) -> PythonResult:
    """Run ``code`` with ``df`` (the given frame), ``pd`` and ``np`` in scope."""
    with tempfile.TemporaryDirectory(prefix="ayd_sandbox_") as tmp:
        tmp_path = Path(tmp)
        frame.to_json(tmp_path / "input.json", orient="table", index=False)
        (tmp_path / "user_code.py").write_text(code, encoding="utf-8")
        wrapper = tmp_path / "wrapper.py"
        wrapper.write_text(_WRAPPER, encoding="utf-8")
        env = {
            k: v for k, v in os.environ.items() if k in {"SYSTEMROOT", "PATH", "TEMP", "TMP", "LANG"}
        }
        env["PYTHONIOENCODING"] = "utf-8"
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(wrapper), tmp],
                cwd=tmp,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout
            if isinstance(out, bytes):
                out = out.decode("utf-8", "replace")
            return PythonResult(False, _clip(out or ""), "", timed_out=True)
        return PythonResult(proc.returncode == 0, _clip(proc.stdout), _clip(proc.stderr))
