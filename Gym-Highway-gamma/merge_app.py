"""
Highway-env MDP γ Explorer — merge-v0
======================================
Streamlit demo showing why merge-v0 is the environment MOST sensitive
to the discount factor γ.

Why merge is uniquely γ-sensitive
----------------------------------
Every other highway-env task lets a reactive (low-γ) agent survive by
responding to immediate threats.  Merging is different:

  1. PREPARATION — the agent must adjust speed *now* to align with a gap
     that won't arrive for several steps.
  2. EXECUTION — the lane change must happen before the merge lane ends.
  3. CONFIRMATION — maintaining speed/position on the main lane.

With γ ≈ 0 the future reward of a successful merge is discounted to ≈ 0, so
there is no gradient pushing the agent to prepare.  It either crashes
immediately into a gap-less traffic wall, or sits in the merge lane until
the road literally ends.

With γ ≈ 0.9 the agent sees the full return of a clean merge sequence and
learns to slow-or-accelerate now in order to slot into a gap later.

The effective planning depth needed is:
  gap-alignment delay  ≈  4-8 steps   →  requires  γ > 1 - 1/8 ≈ 0.88
This is higher than for highway (where γ > 0.5 is often sufficient).

Run with:
    streamlit run merge_app.py
"""

import io, time, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import streamlit as st
from PIL import Image

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="MDP γ Explorer · Merge",
    page_icon="🔀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Fraunces:opsz,wght@9..144,300;9..144,600&display=swap');
html, body, [class*="css"] { font-family: 'JetBrains Mono', monospace; }
#MainMenu, footer, header { visibility: hidden; }
.app-title {
    font-family: 'Fraunces', Georgia, serif;
    font-size: 2rem; font-weight: 600; color: #1e3a5f;
    letter-spacing: -0.5px; margin-bottom: 0; line-height: 1.2;
}
.app-subtitle {
    font-size: 0.75rem; color: #64748b; letter-spacing: 2px;
    text-transform: uppercase; margin-top: 4px; margin-bottom: 24px;
}
.teach-box {
    background: #0f172a; border-left: 3px solid #60a5fa;
    border-radius: 0 8px 8px 0; padding: 14px 18px;
    font-size: 0.8rem; line-height: 1.75; color: #94a3b8; margin-top: 8px;
}
.warn-box {
    background: #0f172a; border-left: 3px solid #f59e0b;
    border-radius: 0 8px 8px 0; padding: 14px 18px;
    font-size: 0.8rem; line-height: 1.75; color: #94a3b8; margin-top: 8px;
}
.section-hdr {
    font-size: 0.65rem; font-weight: 700; letter-spacing: 2px;
    text-transform: uppercase; color: #475569; margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

_CMAP = LinearSegmentedColormap.from_list(
    "vf", ["#1e3a5f", "#2563eb", "#60a5fa", "#fde68a", "#f59e0b"]
)
ACTION_NAMES  = ["Lane Left", "Idle", "Lane Right", "Faster", "Slower"]
ACTION_COLORS = ["#3b82f6", "#64748b", "#8b5cf6", "#10b981", "#ef4444"]


# ══════════════════════════════════════════════════════════════════════════════
#  Environment helpers
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def import_highway():
    import gymnasium
    import highway_env  # noqa: F401
    return gymnasium


def make_env(gymnasium, vehicles: int, duration: int = 20):
    """
    merge-v0 config notes:
    - merging_speed_reward penalises merging at high speed, encouraging the
      agent to slow into a gap rather than force through at full speed.
    - collision_reward is the dominant negative signal; its interaction with γ
      is what makes the environment so γ-sensitive.
    """
    return gymnasium.make(
        "merge-v0",
        render_mode="rgb_array",
        config={
            "vehicles_count":      vehicles,
            "duration":            duration,
            "observation":         {"type": "TimeToCollision", "horizon": 10},
            "action":              {"type": "DiscreteMetaAction"},
            "policy_frequency":    1,
            "simulation_frequency": 5,
            "show_trajectories":   False,
            "render_agent":        True,
            "collision_reward":    -1,
            "merging_speed_reward": -0.5,
            "high_speed_reward":    0.2,
            "right_lane_reward":    0.1,
            "lane_change_reward":  -0.05,
        },
    )


def make_finite_mdp(env):
    """
    merge-v0 (like roundabout-v0) does not expose to_finite_mdp(), so we
    call finite_mdp() from highway_env.envs.common.finite_mdp directly.
    The merge lane is represented as one of the lane segments; TTC is computed
    against main-lane traffic, making the collision grid especially rich near
    the merge point.
    """
    from highway_env.envs.common.finite_mdp import finite_mdp as _finite_mdp
    unwrapped = env.unwrapped
    tf = 1.0 / unwrapped.config.get("policy_frequency", 1)
    hz = float(unwrapped.config.get("duration", 20))
    return _finite_mdp(unwrapped, time_quantization=tf, horizon=hz)


# ══════════════════════════════════════════════════════════════════════════════
#  Value Iteration  (same Bellman update as the highway demo)
# ══════════════════════════════════════════════════════════════════════════════

def value_iteration(mdp, gamma: float, theta: float = 1e-4, max_iter: int = 500):
    T = np.asarray(mdp.transition, dtype=int)
    R = np.asarray(mdp.reward,     dtype=float)
    n_states, n_actions = T.shape
    V = np.zeros(n_states)
    Q = np.zeros((n_states, n_actions))
    for i in range(max_iter):
        Q     = R + gamma * V[T]
        V_new = Q.max(axis=1)
        delta = np.abs(V_new - V).max()
        V     = V_new
        if delta < theta:
            break
    pi = Q.argmax(axis=1)
    return V, pi, i + 1


# ══════════════════════════════════════════════════════════════════════════════
#  MPC rollout
# ══════════════════════════════════════════════════════════════════════════════

def run_episode(env, gamma: float, n_steps: int = 40):
    """
    Re-snapshot finite_mdp() + re-solve VI at every step (MPC).

    This is especially important for merge because the traffic gap the agent
    is aiming for moves every step.  A stale TTC grid would cause the agent
    to aim for a gap that has already closed.

    env must already be reset by the caller.
    """
    frames       = []
    total_reward = 0.0

    frame = env.render()
    if frame is not None:
        frames.append(frame)

    for _ in range(n_steps):
        try:
            mdp_now = make_finite_mdp(env)
        except Exception:
            break
        _, pi_now, _ = value_iteration(mdp_now, gamma, theta=1e-3, max_iter=150)
        state  = max(0, min(int(mdp_now.state), len(pi_now) - 1))
        action = int(pi_now[state])

        _, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        if done or truncated:
            break

    return frames, total_reward


def frames_to_gif(frames, fps: int = 8) -> bytes:
    if not frames:
        return b""
    pil_frames = [Image.fromarray(f.astype(np.uint8)) for f in frames]
    buf = io.BytesIO()
    pil_frames[0].save(buf, format="GIF", save_all=True,
                       append_images=pil_frames[1:], loop=0,
                       duration=int(1000 / fps), optimize=False)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
#  Plot helpers
# ══════════════════════════════════════════════════════════════════════════════

_DARK = "#0f172a"

def plot_value_function(V):
    fig, ax = plt.subplots(figsize=(5, 2.8))
    fig.patch.set_facecolor(_DARK); ax.set_facecolor(_DARK)
    im = ax.imshow(V.reshape(1, -1), aspect="auto", cmap=_CMAP,
                   vmin=V.min(), vmax=V.max())
    plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.35,
                 fraction=0.08, label="V(s)")
    ax.set_yticks([])
    ax.set_xlabel("State index", color="#94a3b8", fontsize=8)
    ax.set_title("State-Value Function  V(s)", color="#e2e8f0", fontsize=9, pad=6)
    ax.tick_params(colors="#64748b", labelsize=7)
    for s in ax.spines.values(): s.set_edgecolor("#1e293b")
    fig.tight_layout(); return fig


def plot_policy(pi):
    counts = [int(np.sum(pi == a)) for a in range(len(ACTION_NAMES))]
    fig, ax = plt.subplots(figsize=(5, 2.8))
    fig.patch.set_facecolor(_DARK); ax.set_facecolor(_DARK)
    bars = ax.bar(ACTION_NAMES, counts, color=ACTION_COLORS,
                  edgecolor=_DARK, linewidth=1.5, zorder=3)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts, default=1) * 0.02,
                str(cnt), ha="center", va="bottom",
                color="#94a3b8", fontsize=8)
    ax.set_title("Optimal Policy — States per Action",
                 color="#e2e8f0", fontsize=9, pad=6)
    ax.set_ylabel("# states", color="#94a3b8", fontsize=8)
    ax.tick_params(axis="x", colors="#94a3b8", labelsize=7, rotation=15)
    ax.tick_params(axis="y", colors="#64748b", labelsize=7)
    ax.grid(axis="y", color="#1e293b", linewidth=0.7, zorder=0)
    for s in ax.spines.values(): s.set_edgecolor("#1e293b")
    fig.tight_layout(); return fig


def plot_gamma_sweep(sweep_results):
    gammas  = [r["gamma"]  for r in sweep_results]
    rewards = [r["reward"] for r in sweep_results]
    iters   = [r["iters"]  for r in sweep_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 2.8))
    sweep_colors = ["#ef4444","#f97316","#f59e0b","#84cc16","#10b981","#60a5fa"]

    for ax in (ax1, ax2):
        fig.patch.set_facecolor(_DARK); ax.set_facecolor(_DARK)
        ax.tick_params(colors="#64748b", labelsize=7)
        ax.grid(axis="y", color="#1e293b", linewidth=0.7)
        for s in ax.spines.values(): s.set_edgecolor("#1e293b")

    ax1.bar([str(g) for g in gammas], rewards,
            color=sweep_colors, edgecolor=_DARK, linewidth=1.2)
    ax1.set_title("Episode Reward vs γ", color="#e2e8f0", fontsize=9, pad=6)
    ax1.set_xlabel("γ", color="#94a3b8", fontsize=8)
    ax1.set_ylabel("Total Reward", color="#94a3b8", fontsize=8)

    ax2.bar([str(g) for g in gammas], iters,
            color="#6366f1", edgecolor=_DARK, linewidth=1.2)
    ax2.set_title("VI Iterations to Converge vs γ",
                  color="#e2e8f0", fontsize=9, pad=6)
    ax2.set_xlabel("γ", color="#94a3b8", fontsize=8)
    ax2.set_ylabel("Iterations", color="#94a3b8", fontsize=8)

    fig.tight_layout(); return fig


# ══════════════════════════════════════════════════════════════════════════════
#  Teaching note (merge-specific)
# ══════════════════════════════════════════════════════════════════════════════

def teaching_note(gamma: float, iters: int, reward: float,
                  dominant_action: str) -> str:
    horizon = f"≈ {1/(1-gamma):.1f}" if gamma < 1.0 else "∞"

    # Minimum lookahead needed: gap-alignment takes ~6 steps, so
    # γ^6 must still be significant → γ > 0.75 roughly.
    if gamma < 0.25:
        personality = "🔴 **Blind to the merge**"
        body = (
            f"At γ={gamma:.2f} the agent discounts a reward 6 steps away by "
            f"{gamma**6:.3f}×.  It cannot see past the next step, so it has "
            f"no gradient pushing it to slow down or speed up to align with a gap.  "
            f"Expect reckless immediate merges (collision) or passive lane-staying "
            f"(ran out of road).  VI converged in **{iters}** iterations — fast "
            f"because the Bellman contraction factor is just {gamma:.2f}."
        )
    elif gamma < 0.60:
        personality = "🟠 **Too short-sighted for merging**"
        body = (
            f"Horizon ≈ {horizon} steps.  The agent begins to sense nearby gaps "
            f"but cannot plan the full preparation-execution sequence.  "
            f"It may merge successfully when a gap happens to be right there, "
            f"but fails when gap alignment requires several preparatory steps.  "
            f"VI: **{iters}** iterations."
        )
    elif gamma < 0.80:
        personality = "🟡 **Borderline for merge**"
        body = (
            f"Horizon ≈ {horizon} steps — close to the ~6-step gap-alignment "
            f"delay.  Performance is inconsistent: the agent succeeds when "
            f"traffic is light but struggles in dense merge scenarios.  "
            f"Notice the policy assigns significant weight to **Slower** — "
            f"the agent is starting to pace itself against gaps.  "
            f"VI: **{iters}** iterations."
        )
    elif gamma < 0.95:
        personality = "🟢 **Competent merger**"
        body = (
            f"Horizon ≈ {horizon} steps.  The agent sees far enough to plan "
            f"speed adjustments well before the merge point.  The dominant "
            f"action is **{dominant_action}** — consistent with deliberate "
            f"gap-seeking rather than reactive behaviour.  "
            f"This is where merge performance diverges most sharply from lower γ.  "
            f"VI: **{iters}** iterations."
        )
    else:
        personality = "🔵 **Master planner**"
        body = (
            f"Horizon ≈ {horizon} steps.  The agent can plan the entire merge "
            f"approach sequence.  VI needed **{iters}** iterations — slow "
            f"contraction (factor = {gamma:.3f}) is the price of long-horizon "
            f"planning.  Near γ=1 the return sum barely converges; "
            f"γ < 1 is a mathematical requirement for non-episodic tasks."
        )

    return (f"**{personality}** — {body}\n\n"
            f"*Episode reward: {reward:.3f} · "
            f"Effective horizon: {horizon} steps*")


# ══════════════════════════════════════════════════════════════════════════════
#  Streamlit UI
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="app-title">🔀 MDP γ Explorer — Merge</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">'
    'Gymnasium Highway-env · merge-v0 · Value Iteration · '
    'Most γ-Sensitive Environment'
    '</div>',
    unsafe_allow_html=True,
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Parameters")
    gamma = st.slider("Discount factor γ", 0.00, 0.999, 0.90, 0.01,
                      format="%.3f")
    horizon_str = f"≈ {1/(1-gamma):.1f} steps" if gamma < 0.999 else "∞"
    st.caption(f"**Effective horizon:** {horizon_str}  (1 / (1−γ))")

    # Informational note — all γ values are valid; low γ is interesting to observe
    if gamma < 0.875:
        st.markdown(
            f'<div class="warn-box">📉 At γ={gamma:.2f} a reward 8 steps away '
            f'is worth only {gamma**8:.3f}× its face value — too little for the '
            f'agent to plan gap-alignment.  This is intentional: run it and watch '
            f'what breaks.</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("### 🏗️ Environment")
    vehicles = st.slider("Vehicles on main lane", 2, 20, 8,
                         help="More vehicles → denser traffic → harder merge")
    duration = st.slider("Duration (s)",          5, 40, 20)
    ep_steps = st.slider("Rollout steps",         10, 60, 35)
    fps      = st.slider("GIF fps",                4, 15,  8)

    st.divider()
    st.markdown("### ⚙️ Value Iteration")
    st.caption("These settings govern the *display* VI.  "
               "The MPC rollout always uses θ=1e-3, 150 iter.")
    theta    = st.select_slider("Convergence θ",
                                options=[1e-2, 1e-3, 1e-4, 1e-5], value=1e-4,
                                format_func=lambda x: f"{x:.0e}")
    max_iter = st.number_input("Max iterations", 50, 2000, 800, 50)

    st.divider()
    run_btn   = st.button("▶  Run Value Iteration", type="primary",
                          use_container_width=True)
    sweep_btn = st.button("🔄  Compare γ ∈ {0, 0.3, 0.5, 0.7, 0.9, 0.99}",
                          use_container_width=True)

# ── Theory expander ────────────────────────────────────────────────────────────
with st.expander("📖 Why merge-v0 is the most γ-sensitive environment",
                 expanded=False):
    st.markdown(r"""
### The merge problem requires multi-step commitment

$$\underbrace{\text{slow down now}}_{\text{immediate cost}} \;\longrightarrow\;
\underbrace{\text{gap opens in } k \text{ steps}}_{\text{delayed signal}} \;\longrightarrow\;
\underbrace{\text{merge safely}}_{\text{future reward}}$$

The return from the merge is discounted by $\gamma^k$.  For a typical gap
alignment delay of $k \approx 6$–$8$ steps:

| γ | γ⁶ | γ⁸ | Planning verdict |
|---|-----|-----|-----------------|
| 0.00 | 0.000 | 0.000 | Blind — can't see the merge |
| 0.50 | 0.016 | 0.004 | Weak signal, random behaviour |
| 0.75 | 0.178 | 0.100 | Borderline |
| 0.90 | 0.531 | 0.430 | Gap-seeking kicks in |
| 0.99 | 0.941 | 0.923 | Near-optimal |

**Why highway-v0 is less sensitive:** the ego is already in flowing traffic.
Staying in a lane and reacting to immediate TTC threats is viable even for a
myopic agent.  The reward signal is dense and stationary.

**Why merge is different:** the merge lane has a *hard deadline*.  The agent
must cross to the main lane before the road ends — a commitment that requires
lookahead.  No amount of reactive behaviour substitutes for planning.

**Finite MDP state reminder:** state = (ego\_speed\_bin, ego\_lane\_segment,
time\_step).  Other vehicles are encoded as a frozen TTC reward grid, not
state.  The MPC rollout re-snapshots and re-solves VI at every step to keep
that grid fresh.
    """)

# ── Main run ───────────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("Importing highway-env …"):
        gymnasium = import_highway()

    status = st.status("Building merge environment …", expanded=True)
    with status:
        st.write("🏗️  Creating `merge-v0` …")
        env = make_env(gymnasium, vehicles=vehicles, duration=duration)
        env.reset()

        st.write("🔢  Converting to finite MDP …")
        try:
            mdp = make_finite_mdp(env)
        except Exception as e:
            st.error(f"finite_mdp() failed on merge-v0: {e}")
            env.close(); st.stop()

        n_states, n_actions = np.asarray(mdp.transition).shape
        st.write(f"    → {n_states} states · {n_actions} actions")

        st.write(f"⚙️  Running Value Iteration  (γ={gamma:.3f}) …")
        t0 = time.time()
        V, pi, n_iter = value_iteration(mdp, gamma,
                                        theta=float(theta),
                                        max_iter=int(max_iter))
        st.write(f"    → Converged in **{n_iter}** iterations "
                 f"({time.time()-t0:.2f} s)")

        st.write("🎬  Rolling out episode (MPC: re-solve VI each step) …")
        frames, total_reward = run_episode(env, gamma, n_steps=ep_steps)
        env.close()
        st.write(f"    → {len(frames)} frames captured · "
                 f"reward = {total_reward:.3f}")
        status.update(label="✅  Done!", state="complete")

    # ── Metrics ────────────────────────────────────────────────────────────────
    counts     = [int(np.sum(pi == a)) for a in range(len(ACTION_NAMES))]
    dom_action = ACTION_NAMES[int(np.argmax(counts))]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("γ (gamma)",         f"{gamma:.3f}")
    c2.metric("VI iterations",     n_iter)
    c3.metric("Episode reward",    f"{total_reward:.3f}")
    c4.metric("Effective horizon", horizon_str)

    note = teaching_note(gamma, n_iter, total_reward, dom_action)
    st.markdown(f'<div class="teach-box">{note}</div>', unsafe_allow_html=True)
    st.divider()

    # ── Plots ──────────────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-hdr">Value Function V(s)</div>',
                    unsafe_allow_html=True)
        st.pyplot(plot_value_function(V), use_container_width=True)
    with col_b:
        st.markdown('<div class="section-hdr">Optimal Policy — Action Distribution</div>',
                    unsafe_allow_html=True)
        st.pyplot(plot_policy(pi), use_container_width=True)

    st.divider()

    # ── Animation ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Episode Animation (MPC rollout)</div>',
                unsafe_allow_html=True)
    if frames:
        gif_bytes = frames_to_gif(frames, fps=fps)
        st.image(gif_bytes,
                 caption=(f"γ={gamma:.3f} · {len(frames)} frames · "
                          f"reward={total_reward:.3f}"),
                 use_container_width=True)
        with st.expander("🖼️  Frame scrubber"):
            idx = st.slider("Frame", 0, len(frames)-1, 0, key="scrub_main")
            st.image(frames[idx], caption=f"Frame {idx+1}/{len(frames)}")
    else:
        st.warning("No frames captured — check virtual display.")

    st.session_state["last_result"] = {
        "gamma": gamma, "V": V, "pi": pi,
        "iters": n_iter, "reward": total_reward,
    }


# ── Gamma sweep ────────────────────────────────────────────────────────────────
if sweep_btn:
    SWEEP_GAMMAS = [0.0, 0.3, 0.5, 0.7, 0.9, 0.99]

    with st.spinner("Importing highway-env …"):
        gymnasium = import_highway()

    st.markdown("### 📊 γ Sweep — same merge environment, six discount factors")
    st.markdown(
        '<div class="warn-box">'
        'Watch the <b>reward cliff</b> around γ = 0.7–0.9.  '
        'Below that threshold the agent cannot plan the gap-alignment sequence.  '
        'Above it, performance jumps sharply — this cliff is steeper here than '
        'in any other highway-env task.'
        '</div>',
        unsafe_allow_html=True,
    )

    prog         = st.progress(0)
    sweep_results = []
    env = make_env(gymnasium, vehicles=vehicles, duration=duration)

    for k, g in enumerate(SWEEP_GAMMAS):
        prog.progress((k+1)/len(SWEEP_GAMMAS),
                      text=f"γ={g}: VI + MPC rollout …")
        env.reset()
        try:
            mdp_s = make_finite_mdp(env)
        except Exception:
            sweep_results.append({"gamma": g, "V": np.zeros(1),
                                   "pi": np.zeros(1, int),
                                   "iters": 0, "reward": 0.0})
            continue
        # Use tighter theta for sweep to ensure fair comparison
        theta_s    = max(1e-5, 1e-3 * (1 - g))   # tighter as γ grows
        max_iter_s = min(2000, int(50 / max(1 - g, 1e-3)))
        V_s, pi_s, it_s = value_iteration(mdp_s, g,
                                           theta=theta_s,
                                           max_iter=max_iter_s)
        _, rew_s = run_episode(env, g, n_steps=ep_steps)
        sweep_results.append({"gamma": g, "V": V_s, "pi": pi_s,
                               "iters": it_s, "reward": rew_s})

    env.close()
    prog.empty()

    st.pyplot(plot_gamma_sweep(sweep_results), use_container_width=True)

    # Side-by-side value functions
    st.markdown('<div class="section-hdr">Value functions — one column per γ</div>',
                unsafe_allow_html=True)
    cols = st.columns(len(SWEEP_GAMMAS))
    vmin = min(r["V"].min() for r in sweep_results)
    vmax = max(r["V"].max() for r in sweep_results)
    for col, r in zip(cols, sweep_results):
        fig, ax = plt.subplots(figsize=(1.4, 3.5))
        fig.patch.set_facecolor(_DARK); ax.set_facecolor(_DARK)
        ax.imshow(r["V"].reshape(-1, 1), aspect="auto", cmap=_CMAP,
                  vmin=vmin, vmax=vmax)
        ax.set_title(f"γ={r['gamma']}", color="#e2e8f0", fontsize=8, pad=4)
        ax.axis("off")
        fig.tight_layout(pad=0.3)
        col.pyplot(fig, use_container_width=True)
        col.caption(f"iters: {r['iters']}\nreward: {r['reward']:.2f}")

    st.session_state["sweep_results"] = sweep_results


# ── Placeholder ────────────────────────────────────────────────────────────────
if not run_btn and not sweep_btn and "last_result" not in st.session_state:
    st.info("👈  Set γ and click **▶ Run Value Iteration** to begin.", icon="🔀")
    st.markdown("""
**What to try:**

1. Run with **γ = 0.25** → watch the agent merge recklessly or get stuck
2. Run with **γ = 0.75** → borderline behaviour, occasional success
3. Run with **γ = 0.90** → deliberate gap-seeking, reliable merge
4. Click **γ Sweep** → see the reward cliff between γ=0.7 and γ=0.9

**Key difference from highway-v0:**
In highway, a reactive (low-γ) agent survives by dodging immediate threats.
In merge, survival *requires* planning ahead — the merge lane ends, and no
amount of last-second reflexes substitutes for having prepared a gap.
    """)
