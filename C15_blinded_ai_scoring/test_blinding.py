"""
Tests for the blinding engine and aggregation pipeline.

Validates that the scoring mechanism works correctly with zero API calls,
using deterministic stub scores to verify:
  - Blinding produces different permutations per task
  - System names are scrubbed from answer text
  - De-anonymisation correctly recovers original system identities
  - Fleiss' kappa and Krippendorff's alpha compute correctly
  - Majority vote and mean aggregation are correct
  - The full pipeline round-trips without data loss
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from .blinding import (
    BlindingManifest,
    blind_answers,
    create_session,
    scrub_system_names,
)
from .aggregator import (
    deblind_and_aggregate,
    fleiss_kappa,
    krippendorff_alpha,
    write_results,
)
from .judges import JudgeScore
from .rubric import TASKS, TASK_INDEX


class TestScrubbing:
    def test_scrubs_bioclaw(self):
        assert "BioClaw" not in scrub_system_names("BioClaw answered correctly")

    def test_scrubs_biomni(self):
        assert "Biomni" not in scrub_system_names("Biomni retrieved the data")

    def test_scrubs_chatgpt(self):
        assert "ChatGPT" not in scrub_system_names("ChatGPT said the answer is X")

    def test_scrubs_gpt55(self):
        text = "gpt-5.5-thinking-extended produced this output"
        cleaned = scrub_system_names(text)
        assert "gpt-5.5" not in cleaned.lower()

    def test_scrubs_pleas(self):
        assert "PLEAS" not in scrub_system_names("PLEAS agent used the tool")

    def test_preserves_neutral_text(self):
        text = "The protein TP53 has accession P04637"
        assert scrub_system_names(text) == text


class TestBlinding:
    def test_produces_three_responses(self):
        answers = {"A": "answer a", "B": "answer b", "C": "answer c"}
        blinded, record = blind_answers("T01", answers, "test-session")
        assert len(blinded) == 3

    def test_labels_are_response_abc(self):
        answers = {"A": "a", "B": "b", "C": "c"}
        blinded, _ = blind_answers("T01", answers, "session")
        labels = {r.label for r in blinded}
        assert labels == {"Response A", "Response B", "Response C"}

    def test_deterministic_with_same_session(self):
        answers = {"X": "x", "Y": "y", "Z": "z"}
        b1, _ = blind_answers("T05", answers, "session-123")
        b2, _ = blind_answers("T05", answers, "session-123")
        assert [r.label for r in b1] == [r.label for r in b2]

    def test_different_tasks_get_different_shuffles(self):
        answers = {"A": "a", "B": "b", "C": "c"}
        session = "same-session"
        shuffles = []
        for tid in ["T01", "T02", "T03", "T04", "T05", "T06",
                     "T07", "T08", "T09", "T10"]:
            blinded, _ = blind_answers(tid, answers, session)
            order = tuple(r.text for r in blinded)
            shuffles.append(order)
        # With 10 tasks, overwhelmingly likely we get > 1 distinct ordering
        assert len(set(shuffles)) > 1

    def test_record_has_correct_mapping(self):
        answers = {"SysA": "a", "SysB": "b", "SysC": "c"}
        _, record = blind_answers("T01", answers, "session-abc")
        assert set(record.mapping.keys()) == {"SysA", "SysB", "SysC"}
        assert set(record.reverse.values()) == {"SysA", "SysB", "SysC"}
        for sys_id, label in record.mapping.items():
            assert record.reverse[label] == sys_id

    def test_rejects_non_three_systems(self):
        with pytest.raises(ValueError, match="Expected 3"):
            blind_answers("T01", {"A": "a", "B": "b"}, "session")


class TestManifest:
    def test_save_and_load_roundtrip(self, tmp_path):
        session = create_session()
        manifest = BlindingManifest(session_id=session, created_utc="2026-06-19T00:00:00Z")
        answers = {"X": "x", "Y": "y", "Z": "z"}
        for task in TASKS[:3]:
            _, record = blind_answers(task.task_id, answers, session)
            manifest.task_records.append(record)

        path = tmp_path / "manifest.json"
        manifest.save(path)
        loaded = BlindingManifest.load(path)

        assert loaded.session_id == session
        assert len(loaded.task_records) == 3
        assert loaded.task_records[0].task_id == "T01"


class TestFleissKappa:
    def test_perfect_agreement(self):
        # 3 raters all agree on score 2 for every subject
        matrix = [[0, 0, 3]] * 10
        k = fleiss_kappa(matrix)
        assert k == pytest.approx(1.0, abs=0.01)

    def test_maximum_disagreement(self):
        # Each rater picks a different score => kappa = -0.5 (3 raters, 3 categories)
        matrix = [[1, 1, 1]] * 10
        k = fleiss_kappa(matrix)
        assert k == pytest.approx(-0.5, abs=0.01)

    def test_partial_agreement(self):
        matrix = [[0, 0, 3], [0, 1, 2], [1, 0, 2], [0, 0, 3]]
        k = fleiss_kappa(matrix)
        assert -1 <= k <= 1


class TestKrippendorffAlpha:
    def test_perfect_agreement(self):
        units = [[2, 2, 2]] * 10
        a = krippendorff_alpha(units)
        assert a == pytest.approx(1.0, abs=0.01)

    def test_handles_missing(self):
        units = [[2, -1, 2], [1, 1, -1]]
        a = krippendorff_alpha(units)
        assert -1 <= a <= 1


class TestAggregation:
    def _make_scores_and_manifest(self):
        session = "test-session-agg"
        manifest = BlindingManifest(session_id=session, created_utc="2026-06-19")
        answers = {"BioClaw": "bc", "Biomni": "bm", "ChatGPT": "cg"}
        scores = []

        for task in TASKS:
            blinded, record = blind_answers(task.task_id, answers, session)
            manifest.task_records.append(record)
            for resp in blinded:
                for judge_id, s in [("j1", 2), ("j2", 2), ("j3", 1)]:
                    scores.append(JudgeScore(
                        judge_id=judge_id,
                        task_id=task.task_id,
                        response_label=resp.label,
                        score=s,
                        rationale="stub",
                    ))
        return scores, manifest

    def test_deblind_recovers_systems(self):
        scores, manifest = self._make_scores_and_manifest()
        results, summaries, _ = deblind_and_aggregate(scores, manifest)
        system_ids = {r.system_id for r in results}
        assert system_ids == {"BioClaw", "Biomni", "ChatGPT"}

    def test_majority_vote(self):
        scores, manifest = self._make_scores_and_manifest()
        results, _, _ = deblind_and_aggregate(scores, manifest)
        # 2 judges say 2, 1 judge says 1 => majority = 2
        for r in results:
            assert r.majority_score == 2

    def test_reliability_computed(self):
        scores, manifest = self._make_scores_and_manifest()
        _, _, reliability = deblind_and_aggregate(scores, manifest)
        assert "fleiss_kappa" in reliability
        assert "krippendorff_alpha" in reliability

    def test_write_results_creates_files(self, tmp_path):
        scores, manifest = self._make_scores_and_manifest()
        results, summaries, reliability = deblind_and_aggregate(scores, manifest)
        write_results(tmp_path, results, summaries, reliability, scores)
        assert (tmp_path / "blinded_task_scores.csv").exists()
        assert (tmp_path / "blinded_system_summary.csv").exists()
        assert (tmp_path / "judge_rationales.json").exists()
        assert (tmp_path / "inter_rater_reliability.json").exists()
        assert (tmp_path / "scoring_report.txt").exists()
