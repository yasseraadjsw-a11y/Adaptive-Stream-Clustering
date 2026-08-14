from __future__ import annotations
import importlib.util, subprocess, sys

if importlib.util.find_spec("pytest") is None:
    print("pytest_not_installed: unit tests skipped; install locked dependencies to run them")
else:
    raise SystemExit(subprocess.run([sys.executable, "-m", "pytest", "-q"]).returncode)
