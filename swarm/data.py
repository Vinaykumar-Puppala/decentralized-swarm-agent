"""Dataset loading, cleaning, profiling and a declarative (no code execution) query tool for agents."""
import io, re
from pathlib import Path
import numpy as np
import pandas as pd

AGGS = {'sum', 'mean', 'median', 'min', 'max', 'count', 'nunique', 'std'}
ARITH = {'+': lambda a, b: a + b, '-': lambda a, b: a - b, '*': lambda a, b: a * b,
         '/': lambda a, b: a / (b.replace(0, np.nan) if hasattr(b, 'replace') else (b or np.nan))}
OPS = {'==', '!=', '>', '>=', '<', '<=', 'in', 'not in', 'contains'}
MAX_ROWS = 50
# pandas' default NA markers minus "None", which business data often uses as a real category (e.g. Discount Band = None).
NA_VALUES = ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN',
             '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null']


def text_columns(df):
    """object or string dtype columns (pandas 3 uses a dedicated 'str' dtype that select_dtypes('object') skips)."""
    return [c for c in df.columns if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])]


def load_table(source, name=None):
    """Load CSV / Excel / Parquet from a path or a file-like object (e.g. a Streamlit upload)."""
    name = (name or getattr(source, 'name', None) or str(source)).lower()
    if isinstance(source, (str, Path)):
        data = source
    else:
        data = io.BytesIO(source.getvalue() if hasattr(source, 'getvalue') else source.read())
    if name.endswith('.csv'):
        return pd.read_csv(data, keep_default_na=False, na_values=NA_VALUES)
    if name.endswith(('.xlsx', '.xlsm', '.xls')):
        return pd.read_excel(data, keep_default_na=False, na_values=NA_VALUES)
    if name.endswith('.parquet'):
        return pd.read_parquet(data)
    raise ValueError(f"Unsupported file type: {name}")


def clean(df):
    """Light, reversible-in-spirit cleaning. Returns (clean_df, notes) where notes list every change made."""
    notes = []
    df = df.copy()
    new_cols = [re.sub(r'\s+', ' ', str(c)).strip() for c in df.columns]
    renamed = [(o, n) for o, n in zip(df.columns, new_cols) if o != n]
    if renamed:
        notes.append(f"Stripped whitespace from column names: {[o for o, _ in renamed]}")
    df.columns = new_cols

    blank = df.isna().all(axis=1)
    if blank.any():
        notes.append(f"Dropped {int(blank.sum())} fully blank rows")
        df = df[~blank].reset_index(drop=True)

    for c in text_columns(df):
        s = df[c]
        stripped = s.map(lambda v: re.sub(r'\s+', ' ', v).strip() if isinstance(v, str) else v)
        n_ws = int((stripped != s).sum() - (s.isna() & stripped.isna()).sum())
        # Unify values that differ only by case: map each lowercase key to its most common spelling.
        lower = stripped.map(lambda v: v.lower() if isinstance(v, str) else v)
        canon = stripped.groupby(lower).agg(lambda x: x.value_counts().index[0])
        unified = lower.map(lambda v: canon.get(v, v) if isinstance(v, str) else v)
        n_case = int((unified != stripped).sum() - (stripped.isna() & unified.isna()).sum())
        if n_ws or n_case:
            notes.append(f"Column '{c}': normalised whitespace in {n_ws} values, unified letter-case in {n_case} values")
        df[c] = unified

    for c in df.columns:
        if 'date' in c.lower() and pd.api.types.is_numeric_dtype(df[c]):
            s = df[c].dropna()
            if len(s) and s.between(20000, 60000).all():  # Excel serial day numbers (1954-2064)
                df[c] = pd.to_datetime(df[c], unit='D', origin='1899-12-30')
                notes.append(f"Column '{c}': converted Excel serial numbers to dates")
    return df, notes


def profile(raw, df, notes, max_cats=25):
    """Compact text profile of the dataset that every agent receives."""
    out = [f"Raw shape: {raw.shape[0]} rows x {raw.shape[1]} cols. After cleaning: {df.shape[0]} rows x {df.shape[1]} cols."]
    if notes:
        out.append("Cleaning applied:\n- " + "\n- ".join(notes))
    dups = int(df.duplicated().sum())
    out.append(f"Exact duplicate rows after cleaning: {dups}")
    nulls = {c: int(n) for c, n in df.isna().sum().items() if n}
    out.append(f"Missing values per column: {nulls or 'none'}")
    out.append("Columns and dtypes: " + ", ".join(f"{c} ({t})" for c, t in df.dtypes.astype(str).items()))
    for c in text_columns(df):
        vc = df[c].value_counts(dropna=True)
        if len(vc) <= max_cats:
            out.append(f"'{c}' values: " + ", ".join(f"{k}={v}" for k, v in vc.items()))
        else:
            out.append(f"'{c}': {len(vc)} distinct values, top: " + ", ".join(f"{k}={v}" for k, v in vc.head(10).items()))
    num = df.select_dtypes(include='number')
    if not num.empty:
        out.append("Numeric summary:\n" + num.describe().T.round(2).to_string())
    for c in df.select_dtypes(include='datetime').columns:
        out.append(f"'{c}' range: {df[c].min().date()} to {df[c].max().date()}")
    out.append("First 5 rows:\n" + df.head(5).to_string(max_colwidth=30))
    return "\n\n".join(out)


class Dataset:
    def __init__(self, raw, name='dataset'):
        self.name = name
        self.raw = raw
        self.df, self.notes = clean(raw)
        self.profile = profile(raw, self.df, self.notes)

    @classmethod
    def from_source(cls, source, name=None):
        name = name or getattr(source, 'name', None) or Path(str(source)).name
        return cls(load_table(source, name), name)

    def query(self, spec):
        return run_query(self.df, spec)


def _col(df, c):
    if c not in df.columns:
        # tolerate case/whitespace differences in what the model typed
        m = {k.lower().strip(): k for k in df.columns}.get(str(c).lower().strip())
        if m is None:
            raise ValueError(f"Unknown column {c!r}. Available: {list(df.columns)}")
        return m
    return c


def run_query(df, spec):
    """Execute a declarative query.

    spec = {
      "filters":  [{"column": "Country", "op": "==", "value": "France"}],
      "derive":   [{"name": "check", "left": "Sales", "op": "-", "right": "COGS"}],  # + - * /; right may be a number
      "group_by": ["Segment"],
      "metrics":  [{"column": "Profit", "agg": "sum"}],            # output column "Profit_sum"
      "ratios":   [{"name": "margin", "numerator": "Profit_sum", "denominator": "Sales_sum"}],
      "sort_by":  "Profit_sum", "ascending": false, "limit": 10
    }
    Also: {"correlation": ["Units Sold", "Profit", ...]} returns a correlation matrix.
    Returns a text table (at most MAX_ROWS rows).
    """
    if not isinstance(spec, dict):
        raise ValueError("data_query must be a JSON object")
    d = df
    for dv in spec.get('derive') or []:
        name, op = dv.get('name'), dv.get('op')
        if not name or op not in ARITH:
            raise ValueError(f"derive needs a name and op in {sorted(ARITH)}")
        left = d[_col(d, dv.get('left'))]
        r = dv.get('right')
        right = r if isinstance(r, (int, float)) else d[_col(d, r)]
        if d is df:
            d = d.copy()
        d[name] = ARITH[op](left, right)
    for f in spec.get('filters') or []:
        c, op, v = _col(d, f.get('column')), f.get('op', '=='), f.get('value')
        if op not in OPS:
            raise ValueError(f"Unsupported op {op!r}; use one of {sorted(OPS)}")
        s = d[c]
        if pd.api.types.is_datetime64_any_dtype(s) and op not in ('in', 'not in', 'contains'):
            v = pd.Timestamp(v)
        mask = {'==': lambda: s == v, '!=': lambda: s != v, '>': lambda: s > v, '>=': lambda: s >= v,
                '<': lambda: s < v, '<=': lambda: s <= v, 'in': lambda: s.isin(v if isinstance(v, list) else [v]),
                'not in': lambda: ~s.isin(v if isinstance(v, list) else [v]),
                'contains': lambda: s.astype(str).str.contains(str(v), case=False, na=False)}[op]()
        d = d[mask]

    if spec.get('correlation'):
        cols = [_col(d, c) for c in spec['correlation']]
        return f"Correlation (n={len(d)} rows):\n" + d[cols].corr().round(3).to_string()

    group_by = [_col(d, c) for c in (spec.get('group_by') or [])]
    metrics = spec.get('metrics') or [{'column': None, 'agg': 'count'}]
    named = {}
    for m in metrics:
        agg = str(m.get('agg', 'sum')).lower()
        if agg not in AGGS:
            raise ValueError(f"Unsupported agg {agg!r}; use one of {sorted(AGGS)}")
        if m.get('column') is None:
            named['row_count'] = (d.columns[0], 'size')
        else:
            c = _col(d, m['column'])
            named[f"{c}_{agg}"] = (c, agg)

    if group_by:
        res = d.groupby(group_by, dropna=False).agg(**named).reset_index()
    else:
        res = pd.DataFrame({k: [d[c].agg(a) if a != 'size' else len(d)] for k, (c, a) in named.items()})

    ratio_cols = []
    for r in spec.get('ratios') or []:
        num, den = r.get('numerator'), r.get('denominator')
        if num not in res.columns or den not in res.columns:
            raise ValueError(f"Ratio columns must be metric outputs; available: {list(res.columns)}")
        rn = r.get('name') or f"{num}/{den}"
        res[rn] = (res[num] / res[den].replace(0, np.nan)).round(4)
        ratio_cols.append(rn)

    if spec.get('sort_by'):
        sb = spec['sort_by']
        if sb not in res.columns:
            raise ValueError(f"sort_by must be one of {list(res.columns)}")
        res = res.sort_values(sb, ascending=bool(spec.get('ascending', False)))
    limit = min(int(spec.get('limit') or MAX_ROWS), MAX_ROWS)
    total = len(res)
    res = res.head(limit)
    num_cols = [c for c in res.select_dtypes(include='number').columns if c not in ratio_cols]
    res[num_cols] = res[num_cols].round(2)
    head = f"Query result ({total} rows{', showing ' + str(limit) if total > limit else ''}; {len(d)} source rows after filters):\n"
    return head + res.to_string(index=False)
