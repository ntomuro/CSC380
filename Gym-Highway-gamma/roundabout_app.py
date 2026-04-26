"""
Highway-env MDP γ Explorer — Roundabout-v0
==========================================
Same VI + MPC approach as the highway demo, applied to roundabout-v0.

Key difference from highway: roundabout-v0 has no to_finite_mdp() method,
so we call finite_mdp() from highway_env.envs.common.finite_mdp directly.
The state space is (ego_speed, ego_lane_segment, time_step) — same encoding
as highway, just with roundabout road topology.

Run with:
    streamlit run roundabout_app.py
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
    page_title="MDP γ Explorer · Roundabout",
    page_icon="🔄",
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


# ── Environment helpers ────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def import_highway():
    import gymnasium
    import highway_env  # noqa: F401
    return gymnasium


def make_env(gymnasium, vehicles: int, duration: int = 20):
    return gymnasium.make(
        "roundabout-v0",
        render_mode="rgb_array",
        config={
            "vehicles_count": vehicles,
            "duration": duration,
            "observation": {"type": "TimeToCollision", "horizon": 10},
            "action": {"type": "DiscreteMetaAction"},
            "policy_frequency": 1,
            "simulation_frequency": 5,
            "show_trajectories": False,
            "render_agent": True,
        },
    )


def make_finite_mdp(env):
    """
    roundabout-v0 lacks to_finite_mdp(), so we import finite_mdp() directly.
    The function works on any highway-env AbstractEnv that has IDM vehicles and
    a road network supporting all_side_lanes().
    """
    from highway_env.envs.common.finite_mdp import finite_mdp as _finite_mdp
    unwrapped = env.unwrapped
    tf = 1.0 / unwrapped.config.get("policy_frequency", 1)
    hz = float(unwrapped.config.get("duration", 20))
    return _finite_mdp(unwrapped, time_quantization=tf, horizon=hz)


# ── VI + rollout ───────────────────────────────────────────────────────────────

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


def run_episode(env, gamma: float, n_steps: int = 40):
    """
    MPC rollout: re-snapshot finite_mdp() + re-solve VI at every step.
    The state (ego_speed, ego_lane_segment, time) is read fresh each step,
    so the policy always matches the current traffic configuration.
    env must already be reset by the caller.
    """
    frames = []
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


# ── Plot helpers ───────────────────────────────────────────────────────────────

def plot_value_function(V):
    fig, ax = plt.subplots(figsize=(5, 2.8))
    fig.patch.set_facecolor("#0f172a"); ax.set_facecolor("#0f172a")
    im = ax.imshow(V.reshape(1, -1), aspect="auto", cmap=_CMAP,
                   vmin=V.min(), vmax=V.max())
    plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.35,
                 fraction=0.08, label="V(s)")
    ax.set_yticks([]); ax.set_xlabel("State index", color="#94a3b8", fontsize=8)
    ax.set_title("State-Value Function  V(s)", color="#e2e8f0", fontsize=9, pad=6)
    ax.tick_params(colors="#64748b", labelsize=7)
    for s in ax.spines.values(): s.set_edgecolor("#1e293b")
    fig.tight_layout(); return fig


def plot_policy(pi):
    counts = [int(np.sum(pi == a)) for a in range(len(ACTION_NAMES))]
    fig, ax = plt.subplots(figsize=(5, 2.8))
    fig.patch.set_facecolor("#0f172a"); ax.set_facecolor("#0f172a")
    bars = ax.bar(ACTION_NAMES, counts, color=ACTION_COLORS,
                  edgecolor="#0f172a", linewidth=1.5, zorder=3)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.02,
                str(cnt), ha="center", va="bottom", color="#94a3b8", fontsize=8)
    ax.set_title("Optimal Policy — States per Action", color="#e2e8f0", fontsize=9, pad=6)
    ax.set_ylabel("# states", color="#94a3b8", fontsize=8)
    ax.tick_params(axis="x", colors="#94a3b8", labelsize=7, rotation=15)
    ax.tick_params(axis="y", colors="#64748b", labelsize=7)
    ax.grid(axis="y", color="#1e293b", linewidth=0.7, zorder=0)
    for s in ax.spines.values(): s.set_edgecolor("#1e293b")
    fig.tight_layout(); return fig


# ── UI ─────────────────────────────────────────────────────────────────────────

st.markdown('<div class="app-title">🔄 MDP γ Explorer — Roundabout</div>',
            unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Gymnasium Highway-env · roundabout-v0 · Value Iteration · MPC Rollout</div>',
            unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🎛️ Parameters")
    gamma = st.slider("Discount factor γ", 0.00, 0.999, 0.90, 0.01,
                      format="%.3f")
    horizon_str = f"≈ {1/(1-gamma):.1f} steps" if gamma < 0.999 else "∞"
    st.caption(f"**Effective horizon:** {horizon_str}")

    st.divider()
    st.markdown("### 🏗️ Environment")
    vehicles = st.slider("Vehicles",      2, 20,  8)
    duration = st.slider("Duration (s)",  5, 40, 20)
    ep_steps = st.slider("Rollout steps", 10, 60, 30)
    fps      = st.slider("GIF fps",        4, 15,  8)

    st.divider()
    st.markdown("### ⚙️ Value Iteration (display only)")
    theta    = st.select_slider("Convergence θ",
                                options=[1e-2, 1e-3, 1e-4, 1e-5], value=1e-4,
                                format_func=lambda x: f"{x:.0e}")
    max_iter = st.number_input("Max iterations", 50, 2000, 500, 50)

    st.divider()
    run_btn   = st.button("▶  Run Value Iteration", type="primary",
                          use_container_width=True)
    sweep_btn = st.button("🔄  Compare γ ∈ {0, 0.3, 0.5, 0.7, 0.9, 0.99}",
                          use_container_width=True)

with st.expander("📖 Roundabout vs Highway — what changes?", expanded=False):
    st.markdown(r"""
**Same MDP structure:** state = (ego\_speed\_index, ego\_lane\_segment, time\_step).

**What's different:** in a roundabout, vehicles must yield at entry points and
navigate circular lanes, so the TTC grid tends to be denser and more dynamic
than on a straight highway.  The finite MDP is built with the same
`finite_mdp()` function from `highway_env.envs.common.finite_mdp`, called
directly since `roundabout-v0` does not expose `to_finite_mdp()`.

**Why lower γ often does better here** (same as highway): a far-sighted agent
(γ → 1) plans many steps ahead against a *frozen* TTC snapshot.  Roundabout
traffic rotates quickly, so that snapshot becomes stale faster.
A shorter-horizon agent re-plans (MPC) and adapts sooner.
    """)


# ── Main run ───────────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("Importing highway-env …"):
        gymnasium = import_highway()

    status = st.status("Building environment …", expanded=True)
    with status:
        st.write("🏗️  Creating `roundabout-v0` …")
        env = make_env(gymnasium, vehicles=vehicles, duration=duration)
        env.reset()

        st.write("🔢  Converting to finite MDP …")
        try:
            mdp = make_finite_mdp(env)
        except Exception as e:
            st.error(f"finite_mdp() failed: {e}")
            env.close()
            st.stop()

        n_states, n_actions = np.asarray(mdp.transition).shape
        st.write(f"    → {n_states} states · {n_actions} actions")

        st.write(f"⚙️  Running Value Iteration  (γ={gamma:.3f}) …")
        t0 = time.time()
        V, pi, n_iter = value_iteration(mdp, gamma, theta=float(theta),
                                        max_iter=int(max_iter))
        st.write(f"    → Converged in **{n_iter}** iterations ({time.time()-t0:.2f}s)")

        st.write("🎬  Rolling out episode (MPC: re-solve VI each step) …")
        frames, total_reward = run_episode(env, gamma, n_steps=ep_steps)
        env.close()
        st.write(f"    → {len(frames)} frames · reward = {total_reward:.3f}")
        status.update(label="✅  Done!", state="complete")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("γ (gamma)",         f"{gamma:.3f}")
    c2.metric("VI iterations",     n_iter)
    c3.metric("Episode reward",    f"{total_reward:.3f}")
    c4.metric("Effective horizon", horizon_str)

    st.markdown(
        f'<div class="teach-box">Horizon ≈ {horizon_str}. '
        f'VI converged in <b>{n_iter}</b> iterations. '
        f'Episode reward: <b>{total_reward:.3f}</b>. '
        f'In a roundabout, the optimal policy tends to favour '
        f'<b>{"Idle / Slower" if gamma > 0.6 else "Faster"}</b> — '
        f'{"waiting for a gap" if gamma > 0.6 else "aggressive entry"}.</div>',
        unsafe_allow_html=True,
    )

    st.divider()
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
    st.markdown('<div class="section-hdr">Episode Animation</div>',
                unsafe_allow_html=True)
    if frames:
        gif_bytes = frames_to_gif(frames, fps=fps)
        st.image(gif_bytes,
                 caption=f"γ={gamma:.3f} · {len(frames)} frames · reward={total_reward:.3f}",
                 use_container_width=True)
        with st.expander("🖼️  Frame scrubber"):
            idx = st.slider("Frame", 0, len(frames)-1, 0)
            st.image(frames[idx], caption=f"Frame {idx+1}/{len(frames)}")
    else:
        st.warning("No frames captured.")


# ── Gamma sweep ────────────────────────────────────────────────────────────────
if sweep_btn:
    SWEEP_GAMMAS = [0.0, 0.3, 0.5, 0.7, 0.9, 0.99]
    with st.spinner("Importing highway-env …"):
        gymnasium = import_highway()

    st.markdown("### 📊 γ Sweep")
    prog = st.progress(0)
    sweep_results = []
    env = make_env(gymnasium, vehicles=vehicles, duration=duration)

    for k, g in enumerate(SWEEP_GAMMAS):
        prog.progress((k+1)/len(SWEEP_GAMMAS), text=f"γ={g} …")
        env.reset()
        try:
            mdp_s = make_finite_mdp(env)
        except Exception:
            sweep_results.append({"gamma": g, "V": np.zeros(1),
                                   "pi": np.zeros(1, int), "iters": 0, "reward": 0})
            continue
        V_s, pi_s, it_s = value_iteration(mdp_s, g, theta=1e-3, max_iter=300)
        _, rew_s = run_episode(env, g, n_steps=ep_steps)
        sweep_results.append({"gamma": g, "V": V_s, "pi": pi_s,
                               "iters": it_s, "reward": rew_s})

    env.close(); prog.empty()

    gammas  = [r["gamma"]  for r in sweep_results]
    rewards = [r["reward"] for r in sweep_results]
    iters   = [r["iters"]  for r in sweep_results]
    sweep_colors = ["#ef4444","#f97316","#f59e0b","#84cc16","#10b981","#60a5fa"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 2.8))
    for ax in (ax1, ax2):
        fig.patch.set_facecolor("#0f172a"); ax.set_facecolor("#0f172a")
        ax.tick_params(colors="#64748b", labelsize=7)
        ax.grid(axis="y", color="#1e293b", linewidth=0.7)
        for s in ax.spines.values(): s.set_edgecolor("#1e293b")
    ax1.bar([str(g) for g in gammas], rewards, color=sweep_colors,
            edgecolor="#0f172a", linewidth=1.2)
    ax1.set_title("Episode Reward vs γ", color="#e2e8f0", fontsize=9, pad=6)
    ax1.set_xlabel("γ", color="#94a3b8", fontsize=8)
    ax1.set_ylabel("Total Reward", color="#94a3b8", fontsize=8)
    ax2.bar([str(g) for g in gammas], iters, color="#6366f1",
            edgecolor="#0f172a", linewidth=1.2)
    ax2.set_title("VI Iterations vs γ", color="#e2e8f0", fontsize=9, pad=6)
    ax2.set_xlabel("γ", color="#94a3b8", fontsize=8)
    ax2.set_ylabel("Iterations", color="#94a3b8", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

if not run_btn and not sweep_btn and "last_result" not in st.session_state:
    st.info("👈  Set γ and click **▶ Run Value Iteration** to begin.", icon="🔄")
