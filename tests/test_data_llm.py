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
