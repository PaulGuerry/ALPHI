"""Minimal tabulate (pipe tables) used if the tabulate package cannot be installed."""


def _fmt(v, floatfmt):
    if isinstance(v, float):
        return format(v, floatfmt.lstrip(".") and floatfmt or "g") if floatfmt else f"{v:g}"
    return str(v)


def tabulate(data, headers=(), tablefmt="pipe", showindex=False, floatfmt="g", **kw):
    try:
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            cols = list(data.columns)
            rows = data.values.tolist()
            if showindex is True or (showindex == "default" and False):
                cols, rows = [""] + cols, [[i] + r for i, r in zip(data.index, rows)]
            heads = cols if headers == "keys" else list(headers)
            data = rows
        elif data and isinstance(data[0], dict):
            heads = list(data[0].keys()) if headers == "keys" else list(headers)
            data = [[r.get(h, "") for h in heads] for r in data]
        else:
            heads = list(headers) if headers != "keys" else []
    except Exception:
        heads = list(headers) if headers != "keys" else []
    rows = [[_fmt(v, floatfmt) for v in r] for r in data]
    heads = [str(h) for h in heads]
    w = [max([len(h)] + [len(r[i]) for r in rows]) if heads else max(len(r[i]) for r in rows)
         for i, h in enumerate(heads or rows[0])]
    line = lambda r: "| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(r)) + " |"
    out = [line(heads), "|" + "|".join("-" * (x + 2) for x in w) + "|"] if heads else []
    return "\n".join(out + [line(r) for r in rows])
