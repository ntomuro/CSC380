"""
LangGraph state schema for the prompt injection detection system.

DetectionState is passed between every node in the graph.  Each agent reads
from it and writes only to its own output field(s).  The 'errors' field uses
an Annotated reducer so that parallel branches can both write to it without
one overwriting the other.
"""

from __future__ import annotations
import operator
from typing import Annotated, Optional, TypedDict


class DetectionState(TypedDict):
    # ── input ──────────────────────────────────────────────────────────────
    conversation_id: str
    conversation: list[dict]         # list of {"role": str, "content": str}
    ground_truth_label: Optional[str]

    # ── agent outputs ──────────────────────────────────────────────────────
    intent_result: Optional[dict]        # populated by intent_analysis_node
    hierarchy_result: Optional[dict]     # populated by instruction_hierarchy_node
    risk_result: Optional[dict]          # populated by risk_classification_node

    # ── orchestration metadata ─────────────────────────────────────────────
    # operator.add concatenates lists from parallel branches instead of
    # overwriting — critical when two nodes run concurrently.
    errors: Annotated[list[str], operator.add]
    processing_stage: str
