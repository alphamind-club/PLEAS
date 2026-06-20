"""
C14 — LangGraph F1/F5 Compression Middleware
=============================================
Extracted from: bioplease/agent/a1.py (lines 4875–4905)

These two methods ARE the "compression middleware" described in the paper.
They are called at the end of every phase node (plan, learn, execute, assess)
before the next phase reads state — enforcing:
  • F5 (context compression): _roll_memory() summarises prior phase output
    into long_term_summary before the next phase reads state.
  • F1 (phase isolation): _prune_history() drops all but the last K messages
    from active context, preventing cross-phase bleed.

NOTE: This middleware does NOT enforce F2.
F2 (schema validation before tool calls) is enforced by Pydantic schemas in
bioplease/agent/pleas.py (EvidenceSchema, PlanStepSchema) — see C11 tests
IT-F2-01 through IT-F2-06 for direct evidence.

Total core logic: ~18 lines (excluding blank lines and docstrings).
"""

# ── Extracted verbatim from a1.py ───────────────────────────────────────────

def _roll_memory(self, state, phase_tag: str):
    """F5: Compress previous phase output into long_term_summary before next phase reads."""
    self._ensure_mem(state)
    mem = state["artifacts"]["memory"]
    prev = mem.get("long_term_summary", "")
    msgs = [m for m in state.get("messages", []) if not isinstance(m, SystemMessage)]
    try:
        mem["long_term_summary"] = self._summarize_messages_for_long_term(prev, msgs)
    except Exception as e:
        print(f"[WARNING] Memory summarization failed: {e}. Keeping previous summary.")
    mem["events"].append({
        "t": time.time(),
        "phase": phase_tag,
        "note": f"Rolled memory at {phase_tag}",
    })

def _prune_history(self, state):
    """F1: Keep only the last K messages in active context; rest live in long_term_summary."""
    K = self.short_window
    msgs = state.get("messages", [])
    sys_msgs = [m for m in msgs if isinstance(m, SystemMessage)]
    non_sys  = [m for m in msgs if not isinstance(m, SystemMessage)]
    state["messages"] = sys_msgs + non_sys[-K:]


# ── How it is wired into every phase node ───────────────────────────────────
#
# At the END of the plan() node (and learn, execute, assess identically):
#
#     state["messages"].append(AIMessage(content=msg))
#     self._roll_memory(state, "PLAN")   # ← F5: compress to long_term_summary
#     self._prune_history(state)         # ← F1: drop old messages from RAM
#     state["next_step"] = "learn"
#     return state
#
# This two-line call is inserted into every phase transition.
# The next phase receives:
#   - system prompt (fresh)
#   - long_term_summary (compressed history)
#   - last K messages only (short window)
#   NOT the full raw output of all prior phases.


# ── LangGraph StateGraph wiring ─────────────────────────────────────────────
# Extracted from a1.py lines 6710–6765
#
# workflow = StateGraph(AgentState)
# workflow.add_node("plan",           plan)
# workflow.add_node("learn",          learn)
# workflow.add_node("execute",        execute)
# workflow.add_node("mini_share",     mini_share)
# workflow.add_node("assess",         assess)
# workflow.add_node("share",          share)
# workflow.add_node("overview_assess",self.overview_assess)
#
# workflow.add_edge(START, "plan")
# workflow.add_edge("plan",       "learn")
# workflow.add_edge("learn",      "execute")
# workflow.add_conditional_edges("execute", routing_function_execute,
#     path_map={"execute":"execute","mini_share":"mini_share","assess":"assess"})
# workflow.add_edge("mini_share", "assess")
# workflow.add_conditional_edges("assess", routing_function_assess,
#     path_map={"plan":"plan","learn":"learn","execute":"execute",
#               "assess":"assess","share":"share"})
# workflow.add_edge("share", END)
# self.app = workflow.compile()
# self.checkpointer = MemorySaver()

# ── Paper claim reconciliation ───────────────────────────────────────────────
# CORRECTED for v21:
#
# Old (incorrect) text: "18-line LangGraph middleware for F2"
# Problem: The middleware (_roll_memory + _prune_history) enforces F1 and F5,
#          NOT F2. F2 (schema validation before tool calls) is implemented via
#          Pydantic in pleas.py and is completely separate from this file.
#
# Corrected v21 text:
#   "a thin compression middleware (_roll_memory, _prune_history, ~18 lines of
#    core logic) inserted at each LangGraph phase transition to enforce F5
#    (context compression into long_term_summary) and F1 (active context pruned
#    to the last K messages). F2 schema validation is enforced separately via
#    Pydantic schemas (EvidenceSchema, PlanStepSchema) in pleas.py."
#
# Evidence pointers:
#   C14 (this file)  → F1/F5 middleware, verbatim from a1.py lines 4875–4905
#   C11 IT-F2-02/03  → F2 Pydantic rejection tests (EvidenceSchema)
#   C11 IT-F2-05/06  → F2 Pydantic rejection tests (PlanStepSchema)
