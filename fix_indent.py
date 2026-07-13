import py_compile
import sys

files = [
    r'F:\Projects\新项目\scripts\diagnose.py',
    r'F:\Projects\新项目\scripts\tuning\showcase_optimal_params.py',
    r'F:\Projects\新项目\scripts\tuning\tune_temperature.py',
    r'F:\Projects\新项目\scripts\tuning\tune_topk.py',
]

for f in files:
    try:
        py_compile.compile(f, doraise=True)
        print(f"OK: {f}")
    except Exception as e:
        print(f"ERROR in {f}: {e}")