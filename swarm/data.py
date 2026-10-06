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


def load_tables(source, name=None):
    """Load a file into {table_name: DataFrame}. Excel workbooks yield one table per non-empty sheet."""
    name = name or getattr(source, 'name', None) or str(source)
    stem = Path(name).stem
    if not name.lower().endswith(('.xlsx', '.xlsm', '.xls')):
        return {stem: load_table(source, name)}
    data = source if isinstance(source, (str, Path)) else io.BytesIO(source.getvalue() if hasattr(source, 'getvalue') else source.read())
    sheets = pd.read_excel(data, sheet_name=None, keep_default_na=False, na_values=NA_VALUES)
    sheets = {k: v for k, v in sheets.items() if not v.dropna(how='all').empty}
    if len(sheets) == 1:
        return {stem: next(iter(sheets.values()))}
    return {f"{stem}/{k}": v for k, v in sheets.items()}


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
        df = df[~blank]

    for c in text_columns(df):
        s = df[c]
        stripped = s.map(lambda v: re.sub(r'\s+', ' ', v).strip() if isinstance(v, str) else v)
        n_ws = int((stripped != s).sum() - (s.isna() & stripped.isna()).sum())
        # Unify values that differ only by case: map each lowercase key to its most common spelling.
        lower = stripped.map(lambda v: v.lower() if isinstance(v, str) else v)
        def _canon(x):
            vc = x.value_counts()   # most common spelling; on ties prefer mixed-case ("France") over "france"/"FRANCE"
            return sorted(vc.index, key=lambda v: (-vc[v], v.islower() or v.isupper()))[0]
        canon = stripped.groupby(lower).agg(_canon)
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
    return df.reset_index(drop=True), notes


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

    @property
    def tables(self):
        return {self.name: self}

    def query(self, spec):
        return run_query(self.df, spec)

    def read_rows(self, spec):
        """-> (text, table_name, row_ids)"""
        text, ids = read_rows(self.df, spec)
        return text, self.name, ids


def _col(df, c):
    if c not in df.columns:
        # tolerate case/whitespace differences in what the model typed
        m = {k.lower().strip(): k for k in df.columns}.get(str(c).lower().strip())
        if m is None:
            raise ValueError(f"Unknown column {c!r}. Available: {list(df.columns)}")
        return m
    return c


def apply_derive_filters(df, spec):
    """Apply spec['derive'] (computed columns) then spec['filters'] and return the resulting frame."""
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
    return d


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
    d = apply_derive_filters(df, spec)

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


ROW_LIMIT = 100


def read_rows(df, spec):
    """Row-level access. spec = {"filters": [...], "derive": [...], "sort_by": col, "ascending": true,
    "columns": [...], "offset": 0, "limit": 50}. Returns (text, row_ids). row_ids are stable ids of the
    cleaned table (column `_row`), so the swarm can track which rows have been looked at."""
    if not isinstance(spec, dict):
        raise ValueError("data_rows must be a JSON object")
    d = apply_derive_filters(df, spec)
    if spec.get('sort_by'):
        d = d.sort_values(_col(d, spec['sort_by']), ascending=bool(spec.get('ascending', True)), kind='stable')
    cols = [_col(d, c) for c in spec['columns']] if spec.get('columns') else list(d.columns)
    offset = max(int(spec.get('offset') or 0), 0)
    limit = max(1, min(int(spec.get('limit') or 50), ROW_LIMIT))
    page = d.iloc[offset:offset + limit]
    out = page[cols].copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime('%Y-%m-%d')
    csv = out.to_csv(index_label='_row', float_format='%.10g')
    if not len(page):
        return f"No rows at offset {offset} ({len(d)} rows match).\n", []
    more = 'no more rows after this page' if offset + limit >= len(d) else f'next page: offset {offset + limit}'
    head = (f"ROWS {offset}-{offset + len(page) - 1} of {len(d)} matching (table has {len(df)} rows; {more}). "
            "CSV, first column _row is the row id:\n")
    return head + csv, [int(i) for i in page.index]


class DataCollection:
    """One or more tables (files / Excel sheets). Duck-types Dataset: .name, .profile, .query(spec)."""

    def __init__(self, tables):
        if not tables:
            raise ValueError("no tables")
        self.tables = dict(tables)            # name -> Dataset
        self.name = ", ".join(self.tables)
        self.profile = self._profile()

    @classmethod
    def from_sources(cls, sources):
        """sources: iterable of (file-like-or-path, filename)."""
        tables = {}
        for src, fname in sources:
            for tname, raw in load_tables(src, fname).items():
                base, n = tname, 2
                while tname in tables:
                    tname, n = f"{base}_{n}", n + 1
                tables[tname] = Dataset(raw, tname)
        return cls(tables)

    def _profile(self):
        parts = []
        if len(self.tables) > 1:
            parts.append(f"{len(self.tables)} SEPARATE TABLES (not joined): {', '.join(repr(t) for t in self.tables)}. "
                         "Every data_query / data_rows call must set \"table\".")
        for t, ds in self.tables.items():
            parts.append(f"===== TABLE '{t}' =====\n{ds.profile}")
        return "\n\n".join(parts)

    def _get(self, name):
        if name is None:
            if len(self.tables) == 1:
                return next(iter(self.tables.values()))
            raise ValueError(f"query must set \"table\"; available: {list(self.tables)}")
        if name in self.tables:
            return self.tables[name]
        m = {k.lower().strip(): v for k, v in self.tables.items()}.get(str(name).lower().strip())
        if m is None:
            raise ValueError(f"Unknown table {name!r}. Available: {list(self.tables)}")
        return m

    def query(self, spec):
        if not isinstance(spec, dict):
            raise ValueError("data_query must be a JSON object")
        spec = dict(spec)
        df = self._get(spec.pop('table', None)).df
        return run_query(df, spec)

    def read_rows(self, spec):
        if not isinstance(spec, dict):
            raise ValueError("data_rows must be a JSON object")
        spec = dict(spec)
        ds = self._get(spec.pop('table', None))
        text, ids = read_rows(ds.df, spec)
        return text, ds.name, ids
