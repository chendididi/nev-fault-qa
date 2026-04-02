"""检索基线评估脚本单测。"""

from __future__ import annotations

import json

import scripts.eval_retrieval as eval_retrieval


def test_parse_k_values_sorted_and_unique():
    values = eval_retrieval._parse_k_values([5, 1, 3, 3, -1, 0])
    assert values == [1, 3, 5]


def test_load_eval_dataset_requires_fields(tmp_path):
    dataset_path = tmp_path / "qa.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "question": "q1",
                    "ground_truth": "a1",
                    "ground_truth_context": "ctx1",
                }
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    payload, meta = eval_retrieval.load_eval_dataset(dataset_path)

    assert len(payload) == 1
    assert meta["dataset_size"] == 1
    assert meta["dataset_path"] == str(dataset_path)


def test_evaluate_retriever_metrics():
    qa_items = [
        {"question": "q1", "ground_truth": "a1", "ground_truth_context": "ctx1"},
        {"question": "q2", "ground_truth": "a2", "ground_truth_context": "ctx2"},
    ]
    relevance_sets = [{"c1"}, {"c3"}]

    retrieval_map = {
        "q1": [{"chunk_id": "c1"}, {"chunk_id": "c2"}],
        "q2": [{"chunk_id": "c2"}, {"chunk_id": "c3"}],
    }

    def _retrieve(query: str, top_k: int):
        return retrieval_map[query][:top_k]

    summary, details = eval_retrieval.evaluate_retriever(
        qa_items=qa_items,
        relevance_sets=relevance_sets,
        retriever_name="mock",
        retrieve_fn=_retrieve,
        retrieve_top_k=2,
        k_values=[1, 2],
    )

    assert summary["retriever"] == "mock"
    assert summary["recall_at"]["1"] == 0.5
    assert summary["recall_at"]["2"] == 1.0
    assert summary["mrr"] == 0.75
    assert details[0]["rank"] == 1
    assert details[1]["rank"] == 2


def test_load_qrels_aligns_with_questions(tmp_path):
    qrels_path = tmp_path / "qrels.json"
    qrels_path.write_text(
        json.dumps(
            [
                {"question": "q1", "relevant_chunk_ids": ["c1", "c2"]},
                {"question": "q2", "relevant_chunk_ids": ["c3"]},
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    relevance_sets, meta = eval_retrieval.load_qrels(qrels_path, questions=["q1", "q2"])

    assert relevance_sets == [{"c1", "c2"}, {"c3"}]
    assert meta["label_source"] == "qrels"
    assert meta["qrels_count"] == 2


def test_evaluate_thresholds_reports_failures():
    metrics = {
        "bm25": {"mrr": 0.4, "hit_rate": 0.6, "recall_at": {"1": 0.4, "3": 0.7}},
        "hybrid": {"mrr": 0.3, "hit_rate": 0.5, "recall_at": {"1": 0.3, "3": 0.5}},
    }
    thresholds = {
        "bm25": {"mrr_min": 0.3, "hit_rate_min": 0.5, "recall_at_min": {"1": 0.3}},
        "hybrid": {"mrr_min": 0.4, "recall_at_min": {"3": 0.6}},
    }

    result = eval_retrieval.evaluate_thresholds(metrics, thresholds)

    assert result["overall_pass"] is False
    assert result["retrievers"]["bm25"]["passed"] is True
    assert result["retrievers"]["hybrid"]["passed"] is False
