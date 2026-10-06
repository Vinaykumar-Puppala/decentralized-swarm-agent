import io
import pandas as pd
import pytest
from langchain_core.messages import AIMessage
from swarm.agent import Decision, extract_json
from swarm.data import Dataset, load_table, run_query
from swarm.llm import LLMConfig, make_llm, response_text


class Upload(io.BytesIO):
    name = 'x.csv'


def test_csv_upload_keeps_none_category_and_cleans():
    raw = load_table(Upload(b" Sales,Country,Band\n10,France,None\n20,  FRANCE ,High\n,,\n30,Germany,\n"))
    ds = Dataset(raw, 'x.csv')
    assert list(ds.df.columns) == ['Sales', 'Country', 'Band']
    assert ds.df.Country.tolist() == ['France', 'France', 'Germany']
    assert ds.df.Band.tolist()[0] == 'None' and pd.isna(ds.df.Band.tolist()[2])
    assert any('blank rows' in n for n in ds.notes)


def test_query_group_ratio_filter_derive():
    df = pd.DataFrame({'Seg': ['a', 'a', 'b'], 'Sales': [10, 30, 50], 'Cost': [5, 5, 60]})
    out = run_query(df, {'derive': [{'name': 'Profit', 'left': 'sales', 'op': '-', 'right': 'Cost'}],
                         'group_by': ['Seg'], 'metrics': [{'column': 'Profit', 'agg': 'sum'}, {'column': 'Sales', 'agg': 'sum'}],
                         'ratios': [{'name': 'margin', 'numerator': 'Profit_sum', 'denominator': 'Sales_sum'}],
                         'sort_by': 'margin'})
    lines = out.splitlines()
    assert '0.75' in lines[2] and lines[2].split()[0] == 'a' and '-0.2' in lines[3]
    assert 'source rows' in run_query(df, {'filters': [{'column': 'Seg', 'op': 'in', 'value': ['b']}]})
    with pytest.raises(ValueError, match='Unknown column'):
        run_query(df, {'group_by': ['nope']})
    with pytest.raises(ValueError, match='Unsupported agg'):
        run_query(df, {'metrics': [{'column': 'Sales', 'agg': 'eval'}]})


def test_financial_sample_profile():
    ds = Dataset.from_source('Financial_Sample_Data.xlsx')
    assert ds.df.shape == (704, 16) and 'Sales' in ds.df.columns
    assert pd.api.types.is_datetime64_any_dtype(ds.df['Date'])
    assert 'None=55' in ds.profile and 'Midmarkets=1' in ds.profile
    assert sorted(ds.df.Country.unique()) == ['Canada', 'France', 'Germany', 'Mexico', 'United States of America']


@pytest.mark.parametrize('text', [
    'Sure! ```json\n{"status": "done", "action": "x"}\n``` hope that helps',
    'prefix {"status": "done", "action": "x"} {"other": 1}',
    '<think>{"bogus": true}</think>{"status": "done", "action": "x"}',
])
def test_extract_json(text):
    assert extract_json(text)['status'] == 'done'


def test_decision_coercion():
    d = Decision.model_validate({'status': 'Finished', 'artifact_to_read': '#12', 'confidence': '1.7',
                                 'artifact_content': {'a': 1}, 'data_query': '{"group_by": ["x"]}', 'board_message': None})
    assert (d.status, d.artifact_to_read, d.confidence, d.artifact_content, d.data_query) == \
           ('done', 12, 1.0, '{"a": 1}', {'group_by': ['x']})


def test_providers_build_correct_clients(monkeypatch):
    monkeypatch.setenv('ANTHROPIC_BASE_URL', 'http://should-not-be-used')
    a = make_llm(LLMConfig('anthropic', 'claude-x', 'k'))
    assert type(a).__name__ == 'ChatAnthropic' and 'api.anthropic.com' in a.anthropic_api_url
    o = make_llm(LLMConfig('local', 'qwen2.5:3b'))
    assert type(o).__name__ == 'ChatOpenAI' and o.openai_api_base == 'http://localhost:11434/v1'
    l = make_llm(LLMConfig('litellm', 'alias', 'sk-1', 'http://proxy:4000', temperature=None))
    assert l.openai_api_base == 'http://proxy:4000' and l.temperature is None
    with pytest.raises(ValueError, match='API key'):
        make_llm(LLMConfig('openai', 'gpt'))
    with pytest.raises(ValueError, match='MODEL'):
        LLMConfig('local', '').validate()


def test_config_from_env(monkeypatch):
    for k, v in {'PROVIDER': 'Anthropic', 'MODEL': 'm', 'ANTHROPIC_API_KEY': 'ak', 'TEMPERATURE': 'none'}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv('API_KEY', raising=False)
    c = LLMConfig.from_env()
    assert (c.provider, c.api_key, c.temperature) == ('anthropic', 'ak', None) and c.public()['api_key'] == '***'


def test_response_text_handles_content_blocks():
    assert response_text(AIMessage(content=[{'type': 'text', 'text': 'a'}, {'type': 'tool_use'}, 'b'])) == 'ab'


def _csv(name, text):
    f = Upload(text.encode()); f.name = name
    return f


def test_collection_multi_file_and_excel_sheets(tmp_path):
    from swarm.data import DataCollection
    xl = tmp_path / 'book.xlsx'
    with pd.ExcelWriter(xl) as w:
        pd.DataFrame({'Country': ['France', 'Spain'], 'Target': [100, 50]}).to_excel(w, sheet_name='targets', index=False)
        pd.DataFrame({'x': [1]}).to_excel(w, sheet_name='other', index=False)
        pd.DataFrame().to_excel(w, sheet_name='empty', index=False)
    dc = DataCollection.from_sources([
        (_csv('sales.csv', 'Country,Sales\nfrance,10\nFrance,20\nSpain,5\n'), 'sales.csv'),
        (str(xl), 'book.xlsx'),
        (_csv('sales.csv', 'Country,Sales\nItaly,1\n'), 'sales.csv'),     # duplicate name gets a suffix
    ])
    assert list(dc.tables) == ['sales', 'book/targets', 'book/other', 'sales_2']   # empty sheet skipped
    assert 'SEPARATE TABLES' in dc.profile and 'join' not in dc.profile.lower().replace('joined', '')
    # tables stay independent: each is queried on its own, and a join request is simply not part of the spec
    assert any(l.split() == ['France', '30'] for l in dc.query({'table': 'sales', 'group_by': ['Country'],
                                                              'metrics': [{'column': 'Sales', 'agg': 'sum'}]}).splitlines())
    assert 'Target_max' in dc.query({'table': 'book/targets', 'metrics': [{'column': 'Target', 'agg': 'max'}]})
    with pytest.raises(ValueError, match='must set'):
        dc.query({'metrics': [{'column': 'Sales', 'agg': 'sum'}]})
    with pytest.raises(ValueError, match='Unknown table'):
        dc.query({'table': 'nope'})
    single = DataCollection.from_sources([(_csv('a.csv', 'v\n1\n2\n'), 'a.csv')])
    assert single.query({'metrics': [{'column': 'v', 'agg': 'sum'}]}).split()[-1] == '3'


def test_read_rows_paging_filters_and_ids():
    from swarm.data import DataCollection
    nl = chr(10)
    f = _csv('t.csv', 'Name,Amt' + nl + ''.join(f'n{i},{i * 10}' + nl for i in range(10)))
    dc = DataCollection.from_sources([(f, 't.csv'), (_csv('u.csv', 'a' + nl + '1' + nl), 'u.csv')])
    text, table, ids = dc.read_rows({'table': 't', 'offset': 0, 'limit': 4})
    assert table == 't' and ids == [0, 1, 2, 3] and 'next page: offset 4' in text and '3,n3,30' in text.splitlines()
    text, _, ids = dc.read_rows({'table': 't', 'offset': 8, 'limit': 4})
    assert ids == [8, 9] and 'no more rows' in text
    _, _, ids = dc.read_rows({'table': 't', 'filters': [{'column': 'amt', 'op': '>=', 'value': 70}], 'sort_by': 'Amt', 'ascending': False})
    assert ids == [9, 8, 7]                                   # ids are original row ids, not positions in the result
    text, _, ids = dc.read_rows({'table': 't', 'columns': ['Amt'], 'limit': 1000})
    assert len(ids) == 10 and text.splitlines()[1] == '_row,Amt'
    assert dc.read_rows({'table': 't', 'offset': 99})[2] == []
    with pytest.raises(ValueError, match='must set'):
        dc.read_rows({'offset': 0})


def test_every_row_of_sample_file_is_readable_in_pages():
    ds = Dataset.from_source('Financial_Sample_Data.xlsx')
    seen = []
    for off in range(0, 704, 100):
        text, _, ids = ds.read_rows({'offset': off, 'limit': 100})
        seen += ids
    assert seen == list(range(704))
    assert 'ROWS 0-99 of 704' in ds.read_rows({'limit': 100})[0]


def test_range_helpers():
    from swarm.workspace import merge_ranges, missing_ranges, to_ranges
    assert to_ranges([5, 3, 4, 9, 3]) == [[3, 5], [9, 9]]
    assert merge_ranges([[0, 4], [3, 8], [20, 21]]) == [[0, 8], [20, 21]]
    assert missing_ranges([[2, 4], [8, 9]], 12) == [[0, 1], [5, 7], [10, 11]]
    assert missing_ranges([[0, 11]], 12) == [] and missing_ranges([], 3) == [[0, 2]]
