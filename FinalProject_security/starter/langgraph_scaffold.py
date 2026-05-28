"""
LangGraph Scaffold — AI Foundations Final Project
==================================================
This file teaches you the LangGraph concepts you need for the project.
Read and run each section in order before filling in the starter notebook.

Concepts covered:
  1. TypedDict state schema
  2. Writing a node function
  3. Building a sequential graph (START → A → END)
  4. Parallel fan-out / fan-in  (START → A + B → C → END)  ← required by project
  5. Conditional edges (routing based on state)
  6. Annotated reducers (merging parallel writes)

You do NOT need the OpenAI API to run sections 1–5.
Section 6 shows a live API call you can adapt for your agents.
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. Imports
# ─────────────────────────────────────────────────────────────────────────────
import json
import operator
from typing import Annotated, Optional, TypedDict

from langgraph.graph import END, START, StateGraph


# =============================================================================
# SECTION 1 — State schema
# =============================================================================
# LangGraph passes a single "state" dictionary through every node.
# You declare its shape with a TypedDict so Python and LangGraph both
# know what fields exist and what types they carry.
#
# Rules:
#  • Fields that aren't filled yet should be Optional[...] (start as None).
#  • Fields that multiple parallel nodes both write to need an
#    Annotated[list[T], reducer] so one branch can't clobber the other.

class ReviewState(TypedDict):
    # ── input ──────────────────────────────────────────────────────────────
    text: str                           # the document being reviewed

    # ── per-node outputs ───────────────────────────────────────────────────
    sentiment_result: Optional[dict]    # filled by sentiment_node
    keywords_result:  Optional[dict]    # filled by keywords_node
    summary_result:   Optional[dict]    # filled by summary_node

    # ── shared error log with a reducer ────────────────────────────────────
    # WHY Annotated[list[str], operator.add]?
    # When sentiment_node and keywords_node run IN PARALLEL, LangGraph
    # tries to merge their return dicts.  If both return {"errors": [...]}
    # with a plain list, one will silently overwrite the other.
    # operator.add tells LangGraph to *concatenate* the lists instead.
    errors: Annotated[list[str], operator.add]

    # ── routing flag ───────────────────────────────────────────────────────
    needs_summary: bool                 # set by the router, used by conditional edge


# =============================================================================
# SECTION 2 — Node functions
# =============================================================================
# A node is just a Python function that:
#   • Accepts the full state dict as its only argument.
#   • Returns a PARTIAL dict — only the keys it wants to update.
#     (LangGraph merges this partial dict into the state; untouched keys
#      keep their current values.)

def sentiment_node(state: ReviewState) -> dict:
    """Fake sentiment analysis — replace with a real LLM call in your project."""
    text = state["text"].lower()
    if any(w in text for w in ["great", "good", "excellent", "love"]):
        label, score = "positive", 0.9
    elif any(w in text for w in ["bad", "terrible", "awful", "hate"]):
        label, score = "negative", 0.85
    else:
        label, score = "neutral", 0.6

    # Return ONLY the keys this node updates.
    return {
        "sentiment_result": {"label": label, "score": score},
        "errors": [],       # empty list; operator.add will concat with keywords' errors
    }


def keywords_node(state: ReviewState) -> dict:
    """Fake keyword extraction — replace with a real LLM call in your project."""
    words = state["text"].split()
    # Pretend the top 3 longest words are "key terms".
    keywords = sorted(set(words), key=len, reverse=True)[:3]

    return {
        "keywords_result": {"keywords": keywords},
        "errors": [],
    }


def router_node(state: ReviewState) -> dict:
    """Decides whether a summary step is needed based on text length."""
    needs = len(state["text"].split()) > 10
    return {"needs_summary": needs}


def summary_node(state: ReviewState) -> dict:
    """Fake summarizer — only runs when needs_summary is True."""
    word_count = len(state["text"].split())
    return {
        "summary_result": {
            "summary": f"[Summarized {word_count}-word text]",
            "sentiment": state["sentiment_result"]["label"],
            "top_keywords": state["keywords_result"]["keywords"],
        },
        "errors": [],
    }


def skip_summary_node(state: ReviewState) -> dict:
    """Placeholder node for the 'no summary needed' branch."""
    return {"summary_result": None, "errors": []}


# =============================================================================
# SECTION 3 — Sequential graph  (START → sentiment → END)
# =============================================================================
# The simplest possible graph: one node, two special markers.

def build_sequential_graph():
    builder = StateGraph(ReviewState)

    # Step 1: register nodes (name → function)
    builder.add_node("sentiment", sentiment_node)

    # Step 2: wire up edges
    builder.add_edge(START, "sentiment")    # graph starts here
    builder.add_edge("sentiment", END)      # graph ends here

    # Step 3: compile — this validates the graph and returns a runnable object
    return builder.compile()


def demo_sequential():
    print("=" * 60)
    print("SECTION 3 — Sequential graph")
    print("=" * 60)

    graph = build_sequential_graph()

    # Print the Mermaid diagram to visualise the topology
    print(graph.get_graph().draw_mermaid())

    initial_state = {
        "text": "This product is great!",
        "sentiment_result": None,
        "keywords_result":  None,
        "summary_result":   None,
        "errors": [],
        "needs_summary": False,
    }

    final_state = graph.invoke(initial_state)
    print("sentiment_result:", final_state["sentiment_result"])
    print("errors          :", final_state["errors"])
    print()


# =============================================================================
# SECTION 4 — Parallel fan-out / fan-in  ← THIS IS WHAT YOUR PROJECT REQUIRES
# =============================================================================
#
#   START ──┬──> sentiment_node ──────────┬──> summary_node ──> END
#           └──> keywords_node  ──────────┘
#
# How it works:
#   • Two add_edge(START, ...) calls = START fires BOTH nodes simultaneously.
#   • Two add_edge(..., "summary") calls = summary waits for BOTH to finish
#     before it is scheduled (built-in fan-in behaviour in LangGraph).
#
# In your project: replace "sentiment" → "intent_analysis",
#                           "keywords" → "instruction_hierarchy",
#                           "summary"  → "risk_classification".

def build_parallel_graph():
    builder = StateGraph(ReviewState)

    builder.add_node("sentiment",    sentiment_node)
    builder.add_node("keywords",     keywords_node)
    builder.add_node("summary",      summary_node)

    # Fan-out: START dispatches both parallel nodes
    builder.add_edge(START, "sentiment")
    builder.add_edge(START, "keywords")

    # Fan-in: summary waits for BOTH sentiment AND keywords
    builder.add_edge("sentiment", "summary")
    builder.add_edge("keywords",  "summary")

    builder.add_edge("summary", END)

    return builder.compile()


def demo_parallel():
    print("=" * 60)
    print("SECTION 4 — Parallel fan-out / fan-in")
    print("=" * 60)

    graph = build_parallel_graph()
    print(graph.get_graph().draw_mermaid())
    # Expected output:
    #   __start__ --> sentiment
    #   __start__ --> keywords       ← two arrows from start = fan-out
    #   sentiment --> summary
    #   keywords  --> summary        ← two arrows into summary = fan-in
    #   summary   --> __end__

    initial_state = {
        "text": "I love this excellent product. It works great and looks good.",
        "sentiment_result": None,
        "keywords_result":  None,
        "summary_result":   None,
        "errors": [],
        "needs_summary": False,
    }

    final_state = graph.invoke(initial_state)
    print("sentiment_result:", final_state["sentiment_result"])
    print("keywords_result :", final_state["keywords_result"])
    print("summary_result  :", final_state["summary_result"])
    print("errors          :", final_state["errors"])
    print()


# =============================================================================
# SECTION 5 — Conditional edges  (routing)
# =============================================================================
# Sometimes you want a node to send the graph down different paths
# depending on what it computed.  Use add_conditional_edges for this.
#
# You provide:
#   • the source node name
#   • a routing function that reads the state and returns a string key
#   • a mapping from key → destination node name

def routing_function(state: ReviewState) -> str:
    """Return the name of the next node based on state."""
    if state["needs_summary"]:
        return "go_to_summary"
    else:
        return "skip_summary"


def build_conditional_graph():
    builder = StateGraph(ReviewState)

    builder.add_node("sentiment",     sentiment_node)
    builder.add_node("keywords",      keywords_node)
    builder.add_node("router",        router_node)
    builder.add_node("summary",       summary_node)
    builder.add_node("skip_summary",  skip_summary_node)

    # sentiment and keywords run in parallel
    builder.add_edge(START, "sentiment")
    builder.add_edge(START, "keywords")

    # both converge into router
    builder.add_edge("sentiment", "router")
    builder.add_edge("keywords",  "router")

    # router branches: long text → summary, short text → skip_summary
    builder.add_conditional_edges(
        "router",
        routing_function,
        {
            "go_to_summary": "summary",
            "skip_summary":  "skip_summary",
        },
    )

    builder.add_edge("summary",      END)
    builder.add_edge("skip_summary", END)

    return builder.compile()


def demo_conditional():
    print("=" * 60)
    print("SECTION 5 — Conditional edges")
    print("=" * 60)

    graph = build_conditional_graph()
    print(graph.get_graph().draw_mermaid())

    for text, label in [
        ("Short text.",
         "short — should skip summary"),
        ("This is a much longer piece of text with many words that definitely needs a summary.",
         "long — should run summary"),
    ]:
        state = {
            "text": text,
            "sentiment_result": None,
            "keywords_result":  None,
            "summary_result":   None,
            "errors": [],
            "needs_summary": False,
        }
        result = graph.invoke(state)
        print(f"Input ({label}):")
        print(f"  needs_summary  = {result['needs_summary']}")
        print(f"  summary_result = {result['summary_result']}")
        print()


# =============================================================================
# SECTION 6 — Real LLM node (template for your project)
# =============================================================================
# This is the pattern every agent in your project follows.
# Replace the system prompt, required keys, and return dict for each agent.

def make_llm_node_template():
    """
    Returns a LangGraph node function that calls the OpenAI API.
    This is a factory so you can customise the prompt and schema per agent.
    """
    from openai import OpenAI
    client = OpenAI()   # reads OPENAI_API_KEY from environment

    SYSTEM_PROMPT = """You are a text analyst.
Analyse the text and output a JSON object with EXACTLY these fields:
{
  "tone":       "<formal|casual|technical>",
  "confidence": <float 0.0-1.0>,
  "reasoning":  "<one sentence>"
}
Output ONLY the JSON. No markdown, no preamble."""

    REQUIRED_KEYS = {"tone", "confidence", "reasoning"}

    def tone_analysis_node(state: ReviewState) -> dict:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": f"Analyse this text:\n\n{state['text']}"},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},  # guarantees valid JSON output
            )

            raw    = response.choices[0].message.content
            parsed = json.loads(raw)

            # Validate that all required keys are present
            missing = REQUIRED_KEYS - parsed.keys()
            if missing:
                raise ValueError(f"Model response missing keys: {missing}")

            return {
                "sentiment_result": parsed,   # reusing existing field for demo
                "errors": [],
            }

        except Exception as exc:
            # NEVER crash the graph — return an error-flagged state instead
            return {
                "sentiment_result": None,
                "errors": [f"tone_analysis_node: {exc}"],
            }

    return tone_analysis_node


# =============================================================================
# SECTION 7 — Annotated reducer demo
# =============================================================================
# Run this to see the reducer in action without any graph.

def demo_reducer():
    print("=" * 60)
    print("SECTION 7 — Annotated[list, operator.add] in action")
    print("=" * 60)

    # Simulate what LangGraph does when it merges two parallel node returns.
    # Both nodes write to 'errors'; operator.add concatenates the lists.
    branch_a = {"sentiment_result": {"label": "positive", "score": 0.9}, "errors": []}
    branch_b = {"keywords_result": {"keywords": ["apple"]}, "errors": ["keywords_node: timeout"]}

    # LangGraph's merge: for Annotated fields, apply the reducer function
    merged_errors = operator.add(branch_a["errors"], branch_b["errors"])
    print("Branch A errors:", branch_a["errors"])
    print("Branch B errors:", branch_b["errors"])
    print("Merged  errors :", merged_errors)
    print()
    print("With a PLAIN list (no reducer) one branch would overwrite the other.")
    print("If branch_a ran last, you'd see: [] — branch_b's error would be LOST.")
    print()


# =============================================================================
# SECTION 8 — Checklist: mapping scaffold → your project
# =============================================================================
CHECKLIST = """
Mapping this scaffold to your final project
--------------------------------------------
Scaffold               Your project
─────────────────────  ──────────────────────────────
ReviewState            DetectionState
sentiment_node         intent_analysis_node
keywords_node          instruction_hierarchy_node
summary_node           risk_classification_node
"tone" field           "intent_result" field
operator.add reducer   errors field (same pattern)
build_parallel_graph   build_detection_graph
graph.invoke(state)    detection_graph.invoke(state)

Key differences in your project:
  • Node functions call the OpenAI API (see Section 6 above).
  • The state has 8 fields (see Cell 10 in the notebook).
  • The Mermaid diagram in Cell 11 is your proof of parallel execution.
  • You must NOT use add_edge(START, "intent_analysis") and then
    add_edge("intent_analysis", "instruction_hierarchy") — that is sequential.
    Both parallel nodes must have their own edge FROM START.
"""


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    demo_sequential()
    demo_parallel()
    demo_conditional()
    demo_reducer()
    print(CHECKLIST)
