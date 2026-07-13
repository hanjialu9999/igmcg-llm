import ast
import sys

files = [
    r'F:/Projects/新项目/scripts/diagnose.py',
    r'F:/Projects/新项目/scripts/tuning/showcase_optimal_params.py',
    r'F:/Projects/新项目/scripts/tuning/tune_temperature.py',
    r'F:/Projects/新项目/scripts/tuning/tune_topk.py',
]

for f in files:
    try:
        with open(f, 'r', encoding='utf-8') as fp:
            content = fp.read()
        ast.parse(content)
        print(f"OK: {f}")
    except Exception as e:
        print(f"ERROR in {f}: {e}")
        # Show problematic lines
        lines = content.split('\n')
        if hasattr(e, 'lineno'):
            start = max(0, e.lineno - 3)
            end = min(len(lines), e.lineno + 2)
            for i in range(start, end):
                marker = ">>> " if i + 1 == e.lineno else "    "
                print(f"  {marker}{i+1}: {repr(lines[i])}")