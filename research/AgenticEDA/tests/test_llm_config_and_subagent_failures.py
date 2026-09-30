import os


def test_openai_chat_model_receives_dict_model_kwargs(monkeypatch):
    import src.config.config as config

    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(config, "ChatOpenAI", FakeChatOpenAI)
    config.get_settings.cache_clear()
    config.get_chat_model.cache_clear()

    try:
        config.get_chat_model(max_retries=0)
    finally:
        config.get_settings.cache_clear()
        config.get_chat_model.cache_clear()

    assert captured["model_kwargs"] == {}


def test_subagent_fallback_does_not_crash_when_llm_unavailable(monkeypatch):
    import src.config.config as config
    from src.agent.subagent import run_subagent

    def broken_model(*args, **kwargs):
        raise TypeError("argument of type 'NoneType' is not iterable")

    def execute_code(_sid, _code, _timeout, cell_id=None):
        return [{"output_type": "stream", "text": "count    3\nmean     2\n"}], None

    monkeypatch.setattr(config, "get_chat_model", broken_model)

    result = run_subagent(
        hypothesis_id="h1",
        hypothesis_title="Fallback test",
        hypothesis_description="Exercise fallback when model construction fails.",
        relevant_cols=["LV ActivePower (kW)"],
        all_columns=["LV ActivePower (kW)"],
        time_col=None,
        session_id="s1",
        push_event=lambda _sid, _event: None,
        execute_code=execute_code,
        cell_counter=[0],
    )

    assert result.status in {"complete", "failed"}
    assert result.cell_ids
    assert "Described LV ActivePower" in result.finding
