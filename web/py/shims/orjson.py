"""Stand-in for orjson (compiled; not in Pyodide) on top of the json module."""
import json
OPT_INDENT_2 = 1


def _default(o):
    import numpy as np
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")


def loads(s):
    return json.loads(s)


def dumps(obj, option=0, default=None):
    return json.dumps(obj, indent=2 if option & OPT_INDENT_2 else None, ensure_ascii=False,
                      default=default or _default).encode("utf-8")
