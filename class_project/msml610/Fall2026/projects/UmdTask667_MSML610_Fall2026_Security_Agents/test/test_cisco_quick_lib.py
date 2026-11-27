"""Check quick-scan result semantics and local inference caching."""

import json
from types import SimpleNamespace

import pytest

import cisco_quick_lib as quicklib
from runners import cisco_local_completion as local


def test_remediation_is_extracted_without_defence_section():
    text = (
        "6. **Impact**: Data exposure\n"
        "7. **Remediation**: Use parameters.\n"
        "```python\nexecute(sql, args)\n```\n"
        "8. **Defence in Depth**: Restrict database permissions."
    )
    actual = quicklib.extract_remediation(text)
    assert "execute(sql, args)" in actual
    assert "Defence" not in actual
    assert quicklib.extract_remediation("No remediation supplied") == ""


def test_quick_report_retains_findings_even_if_ai_disagrees(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({
        "report_metadata": {"llm_enabled": True},
        "findings": [{
            "file": "/tmp/case_1.py", "start_line": 8,
            "rule_id": "sql-rule", "severity": "ERROR",
            "semgrep_message": "Potential SQL injection",
            "metadata": {"cwe": ["89"], "confidence": "HIGH"},
            "ai_analysis": {"status": "success", "message": "False positive"},
        }],
    }))
    findings = quicklib.read_report(path)
    assert len(findings) == 1
    assert findings[0]["cwe"] == 89
    assert findings[0]["line"] == 8
    assert findings[0]["fix"] == ""


def test_failed_llm_analysis_is_rejected(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({
        "report_metadata": {"llm_enabled": True},
        "findings": [{"ai_analysis": {"status": "timeout"}}],
    }))
    with pytest.raises(ValueError, match="failed LLM"):
        quicklib.read_report(path)


def test_native_token_counters():
    stderr = (
        "llama_perf_context_print: prompt eval time = 2.0 ms / 123 tokens\n"
        "llama_perf_context_print:        eval time = 3.0 ms / 45 runs\n"
    )
    assert quicklib.parse_timings(stderr) == {"tokens_in": 123, "tokens_out": 45}
    assert quicklib.parse_timings("") == {"tokens_in": None, "tokens_out": None}


def test_local_wrapper_uses_model_file_and_reuses_cache(
    tmp_path, monkeypatch, capsys
):
    for name, value in {
        "CISCO_LLAMA_BINARY": "/fake/llama-completion",
        "CISCO_MODEL_FILE": "/fake/pinned.gguf",
        "CISCO_MODEL_SHA256": "hash",
        "CISCO_LOCAL_CACHE": str(tmp_path),
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(local.sys, "argv", [
        "llama-completion", "--hf-repo", "upstream/repo", "--hf-file", "x.gguf",
        "-p", "prompt", "-n", "128",
    ])
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="analysis", stderr="", returncode=0)

    monkeypatch.setattr(local.subprocess, "run", fake_run)
    assert local.main() == 0
    assert local.main() == 0
    assert len(calls) == 1
    assert "--hf-repo" not in calls[0]
    assert calls[0][1:3] == ["--model", "/fake/pinned.gguf"]
    assert capsys.readouterr().out == "analysisanalysis"
