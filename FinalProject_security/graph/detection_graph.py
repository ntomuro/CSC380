"""
LangGraph detection graph.

Topology (parallel fan-out / fan-in):

    START ──┬──> intent_analysis ──────────┬──> risk_classification ──> END
            └──> instruction_hierarchy ────┘

Both Agent 1 and Agent 2 run concurrently.  risk_classification is only
scheduled when BOTH upstream nodes have completed (LangGraph's built-in
fan-in: a node waits for all incoming edges to be satisfied).
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.intent_analysis_agent import intent_analysis_node
from agents.instruction_hierarchy_agent import instruction_hierarchy_node
from agents.risk_classification_agent import risk_classification_node
from agents.state import DetectionState


def build_detection_graph() -> StateGraph:
    builder = StateGraph(DetectionState)

    # Register nodes
    builder.add_node("intent_analysis",       intent_analysis_node)
    builder.add_node("instruction_hierarchy", instruction_hierarchy_node)
    builder.add_node("risk_classification",   risk_classification_node)

    # Fan-out: START fires both parallel agents simultaneously
    builder.add_edge(START, "intent_analysis")
    builder.add_edge(START, "instruction_hierarchy")

    # Fan-in: risk_classification waits for BOTH to finish
    builder.add_edge("intent_analysis",       "risk_classification")
    builder.add_edge("instruction_hierarchy", "risk_classification")

    builder.add_edge("risk_classification", END)

    return builder.compile()


detection_graph = build_detection_graph()


def make_initial_state(conversation: dict) -> DetectionState:
    """Build the initial DetectionState dict from a dataset conversation."""
    return {
        "conversation_id":     conversation["id"],
        "conversation":        conversation["turns"],
        "ground_truth_label":  conversation.get("label"),
        "intent_result":       None,
        "hierarchy_result":    None,
        "risk_result":         None,
        "errors":              [],
        "processing_stage":    "started",
    }
