"""生成基线评估脚本单测。"""

from __future__ import annotations

import json

import scripts.eval_generation_baseline as generation_baseline


def test_load_thresholds(tmp_path):
    path = tmp_path / "thresholds.json"
    path.write_text(
        json.dumps(
            {
                "faithfulness_min": 0.5,
                "answer_relevancy_min": 0.4,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    thresholds = generation_baseline.load_thresholds(path)

    assert thresholds["faithfulness"] == 0.5
    assert thresholds["answer_relevancy"] == 0.4


def test_evaluate_scores():
    scores = {
        "faithfulness": 0.6,
        "answer_relevancy": 0.55,
        "context_precision": 0.4,
        "context_recall": 0.8,
    }
    thresholds = {
        "faithfulness": 0.5,
        "answer_relevancy": 0.6,
    }

    result = generation_baseline.evaluate_scores(scores, thresholds)

    assert result["overall_pass"] is False
    failed = [item for item in result["checks"] if not item["passed"]]
    assert len(failed) == 1
    assert failed[0]["metric"] == "answer_relevancy"


def test_build_ragas_command_contains_required_args(tmp_path):
    parser = generation_baseline.build_parser()
    args = parser.parse_args(
        [
            "--qa-file",
            "tests/eval_qa_set.json",
            "--config",
            "config/config.cpu.yaml",
            "--llm-model",
            "gpt-4o-mini",
        ]
    )
    output_path = tmp_path / "ragas.json"

    cmd = generation_baseline._build_ragas_command(
        args,
        ragas_output=output_path,
        api_base="http://127.0.0.1:8000/api/v1",
    )

    assert cmd[0].endswith("python")
    assert "tests/eval_ragas.py" in cmd
    assert "--qa-file" in cmd
    assert "--output" in cmd
    assert str(output_path) in cmd
    assert "--api-base" in cmd
    assert "--api-timeout" in cmd
    assert "--llm-model" in cmd
