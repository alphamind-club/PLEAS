from bioplease.agent.cost_manager import CostManager


def test_basic_usage():
    cm = CostManager()
    text = "Hello world! This is a small test."
    toks = cm.estimate_tokens(text)
    assert toks > 0
    rec = cm.record_usage("gpt-4o-mini", toks)
    assert rec["model"] == "gpt-4o-mini"
    rpt = cm.get_report()
    assert rpt["total_tokens"] == toks
    assert "gpt-4o-mini" in rpt["by_model"]


if __name__ == "__main__":
    test_basic_usage()
    print("OK")
