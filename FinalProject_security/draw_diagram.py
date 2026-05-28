"""
Generates system_diagram.png
Run: python draw_diagram.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

# ── Canvas ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 24))
ax.set_xlim(0, 9)
ax.set_ylim(0, 24)
ax.axis("off")
BG = "#F7F9FC"
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# ── Colours ───────────────────────────────────────────────────────────────────
C_DATASET = "#1A5276"
C_INPUT   = "#0E6655"
C_LG_BG   = "#EBF5FB"
C_LG_BDR  = "#2471A3"
C_STATE   = "#6C3483"
C_PAR     = "#D4E6F1"
C_AGENT12 = "#1F618D"
C_AGENT3  = "#922B21"
C_OUTPUT  = "#1E8449"
C_EVAL    = "#6E2F0A"
C_NODE    = "#2C3E50"
C_ARROW   = "#2C3E50"
WHITE     = "#FFFFFF"

CX  = 4.5   # horizontal centre
PAD = 0.15  # FancyBboxPatch outer pad

# ── Helpers ───────────────────────────────────────────────────────────────────

def rbox(x, y, w, h, fill, edge=WHITE, lw=2, alpha=1.0, z=3):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={PAD}",
        facecolor=fill, edgecolor=edge,
        linewidth=lw, alpha=alpha, zorder=z,
    ))

def txt(x, y, s, size=10, color=WHITE, weight="bold",
        ha="center", va="center", style="normal", z=4):
    ax.text(x, y, s, ha=ha, va=va,
            fontsize=size, color=color, fontweight=weight,
            fontstyle=style, multialignment="center",
            linespacing=1.4, zorder=z)

def arrow(x1, y1, x2, y2, lw=2.5, color=C_ARROW, rad=0.0, ms=22):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>", color=color, lw=lw,
            mutation_scale=ms,
            connectionstyle=f"arc3,rad={rad}",
        ),
        zorder=7,
    )

def circ(cx, cy, r=0.30, label="", color=C_NODE):
    ax.add_patch(Circle((cx, cy), r, color=color, zorder=6))
    if label:
        txt(cx, cy, label, size=7.5, z=7)


# ═══════════════════════════════════════════════════════════════════════════════
# TITLE  (y ≈ 23.0 – 23.8)
# ═══════════════════════════════════════════════════════════════════════════════
txt(CX, 23.55,
    "Multi-Agent Adversarial Prompt Injection\nDetection System",
    size=12, color="#1C2833")
txt(CX, 22.85,
    "System Architecture  —  AI Foundations Final Project",
    size=8.5, color="#7F8C8D", weight="normal", style="italic")

# ═══════════════════════════════════════════════════════════════════════════════
# DATASET  inner: y 20.9 – 22.3   h = 1.4
# ═══════════════════════════════════════════════════════════════════════════════
rbox(0.6, 20.9, 7.8, 1.4, C_DATASET)
txt(CX, 22.02, "Synthetic Dataset", size=10.5)
txt(CX, 21.58, "60 labelled conversations  —  synthetic_dataset.py",
    size=8, color="#AED6F1", weight="normal")
txt(CX, 21.15, "20 Benign  ·  15 Suspicious  ·  25 Injection",
    size=8.5, color="#F0F3F4", weight="normal")

# Arrow: Dataset → Input
arrow(CX, 20.75, CX, 20.08, lw=3.0, ms=24)

# ═══════════════════════════════════════════════════════════════════════════════
# INPUT  inner: y 18.55 – 19.9   h = 1.35
# ═══════════════════════════════════════════════════════════════════════════════
rbox(0.6, 18.55, 7.8, 1.35, C_INPUT)
txt(CX, 19.55, "Input Conversation", size=10.5)
txt(CX, 19.02,
    "system_prompt  ·  user_messages  ·  tool_descriptions",
    size=8, color="#A9DFBF", weight="normal")

# Arrow: Input → LangGraph
arrow(CX, 18.40, CX, 17.72, lw=3.0, ms=24)

# ═══════════════════════════════════════════════════════════════════════════════
# LANGGRAPH BOX  inner: y 7.6 – 17.55   h = 9.95
#   (bottom raised so it sits snugly around END at y=8.25)
# ═══════════════════════════════════════════════════════════════════════════════
rbox(0.35, 7.6, 8.3, 9.95, C_LG_BG, edge=C_LG_BDR, lw=3, alpha=1.0, z=1)
txt(CX, 17.22, "LangGraph  StateGraph", size=11.5, color=C_LG_BDR, z=2)

# ── DetectionState bar  inner: y 16.12 – 16.78   h = 0.66 ────────────────────
rbox(0.65, 16.12, 7.7, 0.66, C_STATE, z=3)
txt(CX, 16.60, "DetectionState", size=9.5, z=4)
txt(CX, 16.25,
    "conversation · intent_result · hierarchy_result · risk_result · errors",
    size=8, color="#D7BDE2", weight="normal", z=4)

# Arrow: DetectionState → START
arrow(CX, 15.97, CX, 15.38, lw=2.4, ms=20)

# ── START  (centre y = 15.06) ─────────────────────────────────────────────────
circ(CX, 15.06, label="START")

# ── Fan-out arrows: convex bows, to Agent 1 and Agent 2 tops (y ≈ 13.67) ─────
arrow(4.18, 14.76, 2.35, 13.67, lw=2.4, ms=20, rad= 0.25)   # → Agent 1
arrow(4.82, 14.76, 6.65, 13.67, lw=2.4, ms=20, rad=-0.25)   # → Agent 2

# ── Parallel band  inner: y 11.65 – 13.85   h = 2.2 ─────────────────────────
rbox(0.55, 11.65, 7.9, 2.2, C_PAR, edge="#AED6F1", lw=1.5, alpha=0.6, z=2)
txt(CX, 13.70, "Agents 1 & 2  run concurrently",
    size=7.5, color="#1A5276", weight="normal", style="italic", z=4)

# ── Agent 1  inner: y 11.82 – 13.52   x: 0.60 – 4.10   w=3.5  h=1.70 ────────
rbox(0.60, 11.82, 3.50, 1.70, C_AGENT12, z=3)
txt(2.35, 13.30, "Agent 1",                          size=9,   z=4)
txt(2.35, 12.93, "Intent Analysis",                  size=8.5, z=4)
txt(2.35, 12.52, "intent_label:",                    size=7,   color="#AED6F1", weight="normal", z=4)
txt(2.35, 12.11, "aligned | ambiguous | misaligned", size=7.5, z=4)

# ── Agent 2  inner: y 11.82 – 13.52   x: 4.90 – 8.40   w=3.5  h=1.70 ────────
rbox(4.90, 11.82, 3.50, 1.70, C_AGENT12, z=3)
txt(6.65, 13.30, "Agent 2",                          size=9,   z=4)
txt(6.65, 12.93, "Hierarchy Analysis",               size=8.5, z=4)
txt(6.65, 12.52, "override_type:",                   size=7,   color="#AED6F1", weight="normal", z=4)
txt(6.65, 12.11, "direct | indirect | none",         size=7.5, z=4)

# ── Fan-in arrows: Agent bottoms → Agent 3 top (y ≈ 11.0) ────────────────────
arrow(2.35, 11.67, 3.55, 11.00, lw=2.4, ms=20)
arrow(6.65, 11.67, 5.45, 11.00, lw=2.4, ms=20)

# ── Agent 3  inner: y 9.25 – 10.85   x: 1.05 – 7.95   w=6.9  h=1.60 ─────────
#   (moved up by 0.60 to close the gap below the parallel band)
rbox(1.05, 9.25, 6.9, 1.60, C_AGENT3, z=3)
txt(CX, 10.65, "Agent 3",                               size=10,  z=4)
txt(CX, 10.30, "Risk Classification",                   size=9.5, z=4)
txt(CX,  9.88, "verdict:",                              size=8,   color="#F1948A", weight="normal", z=4)
txt(CX,  9.48, "Benign  |  Suspicious  |  Injection",  size=9,   color="#F8C4C4", z=4)

# Arrow: Agent 3 → END
arrow(CX, 9.10, CX, 8.57, lw=2.4, ms=20)

# ── END  (centre y = 8.25) ────────────────────────────────────────────────────
#   (moved up to be snug under Agent 3)
circ(CX, 8.25, label="END")

# ── Arrow: LangGraph bottom → Output ─────────────────────────────────────────
arrow(CX, 7.45, CX, 6.85, lw=3.0, ms=24)

# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT  inner: y 5.5 – 6.7   h = 1.2
# ═══════════════════════════════════════════════════════════════════════════════
rbox(0.6, 5.5, 7.8, 1.2, C_OUTPUT)
txt(CX, 6.38, "Detection Output", size=10.5, z=4)
txt(CX, 5.92,
    "verdict  ·  confidence (0–1)  ·  explanation  ·  contributing signals",
    size=7.5, color="#A9DFBF", weight="normal", z=4)

# Arrow: Output → Evaluation
arrow(CX, 5.35, CX, 4.80, lw=3.0, ms=24)

# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION  inner: y 3.5 – 4.65   h = 1.15
# ═══════════════════════════════════════════════════════════════════════════════
rbox(0.6, 3.5, 7.8, 1.15, C_EVAL)
txt(CX, 4.38, "Evaluation Metrics", size=10.5, z=4)
txt(CX, 3.90,
    "Accuracy  ·  Macro F1  ·  Confusion Matrix  ·  Error Analysis (FP/FN)",
    size=7.5, color="#FAD7A0", weight="normal", z=4)

# ═══════════════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════════════
plt.tight_layout(pad=0.3)
plt.savefig("system_diagram.png", dpi=180, bbox_inches="tight", facecolor=BG)
print("Saved: system_diagram.png")
plt.close()
