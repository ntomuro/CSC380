"""
MDP γ Explorer — Parking-v0
============================
Uses actual Gymnasium parking-v0 rendering for all episode animation.

Because parking-v0 has a continuous kinematic state (x, y, vx, vy, heading)
and no to_finite_mdp(), we build a thin 2-D grid MDP whose coordinates are
aligned with the normalised observation space of the real environment:

  normalised x = actual_x / 100  (scale factor from KinematicsGoal config)
  normalised y = actual_y / 100

The grid computes V(s) and π*(s) via Value Iteration.  At each rollout step
the agent's real normalised (x, y) is mapped to a grid cell, the optimal
grid action (UP/DOWN/LEFT/RIGHT) is read from π*, and converted to a
continuous [steering, acceleration] command using a simple heading controller.

Run with:
    streamlit run parking_app.py
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
    page_title="MDP γ Explorer · Parking",
    page_icon="🅿️",
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
_DARK = "#0f172a"


# ══════════════════════════════════════════════════════════════════════════════
#  Grid MDP in normalised coordinate space
#
#  parking-v0 KinematicsGoal observation divides metres by 100.
#  The parking lot fits comfortably in x ∈ [-0.25, 0.25], y ∈ [-0.12, 0.12].
# ══════════════════════════════════════════════════════════════════════════════

GRID_COLS = 20
GRID_ROWS = 10
X_MIN, X_MAX = -0.25, 0.25
Y_MIN, Y_MAX = -0.12, 0.12
CELL_W = (X_MAX - X_MIN) / GRID_COLS
CELL_H = (Y_MAX - Y_MIN) / GRID_ROWS
N_STATES  = GRID_ROWS * GRID_COLS
N_ACTIONS = 4

# UP / DOWN / LEFT / RIGHT  in (Δrow, Δcol)
DELTAS       = [(-1, 0), (+1, 0), (0, -1), (0, +1)]
ACTION_NAMES = ["Up ↑", "Down ↓", "Left ←", "Right →"]


def rc_to_s(r, c):  return r * GRID_COLS + c
def s_to_rc(s):     return divmod(s, GRID_COLS)


def xy_to_rc(x_norm, y_norm):
    """Normalised (x, y) → nearest grid (row, col)."""
    col = int((x_norm - X_MIN) / CELL_W)
    row = int((y_norm - Y_MIN) / CELL_H)
    return (max(0, min(GRID_ROWS - 1, row)),
            max(0, min(GRID_COLS - 1, col)))


def rc_to_xy(r, c):
    """Centre of cell (r, c) in normalised coords."""
    return (X_MIN + (c + 0.5) * CELL_W,
            Y_MIN + (r + 0.5) * CELL_H)


def build_mdp(goal_rc, start_rc):
    """
    Open-field grid MDP: boundary walls only, goal at goal_rc.
    No internal obstacles — the real parking-lot geometry is handled by
    the continuous environment during rollout.
    """
    goal_s = rc_to_s(*goal_rc)
    T = np.zeros((N_STATES, N_ACTIONS), dtype=int)
    R = np.zeros((N_STATES, N_ACTIONS))

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            s = rc_to_s(r, c)
            if s == goal_s:
                for a in range(N_ACTIONS):
                    T[s, a] = goal_s
                    R[s, a] = 0.0
                continue
            for a, (dr, dc) in enumerate(DELTAS):
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= GRID_ROWS or nc < 0 or nc >= GRID_COLS:
                    nr, nc = r, c          # bounce off boundary
                ns     = rc_to_s(nr, nc)
                T[s, a] = ns
                R[s, a] = 10.0 if ns == goal_s else -0.1

    class GridMDP:
        def __init__(self):
            self.transition = T
            self.reward     = R
            self.state      = rc_to_s(*start_rc)

    return GridMDP()


# ══════════════════════════════════════════════════════════════════════════════
#  Value Iteration  (same Bellman update as every other demo in this suite)
# ══════════════════════════════════════════════════════════════════════════════

def value_iteration(mdp, gamma: float, theta: float = 1e-4,
                    max_iter: int = 1000):
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
#  Environment helpers
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def import_highway():
    import gymnasium
    import highway_env  # noqa
    return gymnasium


def make_env(gymnasium):
    return gymnasium.make(
        "parking-v0",
        render_mode="rgb_array",
        config={
            "simulation_frequency": 15,
            "policy_frequency":      5,
            "duration":            100,
            "screen_width":        600,
            "screen_height":       300,
        },
    )


def _extract_obs(obs):
    """Return (ego_vec, goal_vec) from a KinematicsGoal dict or flat array."""
    if isinstance(obs, dict):
        return obs["observation"], obs["desired_goal"]
    return obs[:6], obs[6:12]


# ══════════════════════════════════════════════════════════════════════════════
#  Rollout using actual parking-v0 rendering
# ══════════════════════════════════════════════════════════════════════════════

CLOSE = 0.08   # switch to heading-alignment phase within ~8 m of goal


def run_episode(env, gamma: float, theta: float, max_iter: int,
                n_steps: int = 250):
    """
    Two-phase parking controller:
      Phase 1 (dist > CLOSE): grid policy for long-range navigation.
        The quality of the path depends on γ — lower γ takes myopic shortcuts.
      Phase 2 (dist ≤ CLOSE): heading-alignment controller for final slot-in.
        Speed is allowed only when heading error is small.

    Returns: frames, total_reward, V, pi, n_iter, goal_rc, start_rc
    """
    obs, _ = env.reset()
    ego, goal_obs = _extract_obs(obs)

    start_rc = xy_to_rc(float(ego[0]),  float(ego[1]))
    goal_rc  = xy_to_rc(float(goal_obs[0]), float(goal_obs[1]))

    mdp = build_mdp(goal_rc, start_rc)
    V, pi, n_iter = value_iteration(mdp, gamma, theta=theta, max_iter=max_iter)

    frames       = []
    total_reward = 0.0

    frame = env.render()
    if frame is not None:
        frames.append(frame)

    for _ in range(n_steps):
        ego, goal_obs = _extract_obs(obs)
        x, y         = float(ego[0]),      float(ego[1])
        cos_h, sin_h = float(ego[4]),      float(ego[5])
        heading      = np.arctan2(sin_h, cos_h)

        gx, gy       = float(goal_obs[0]), float(goal_obs[1])
        g_cos, g_sin = float(goal_obs[4]), float(goal_obs[5])
        goal_heading = np.arctan2(g_sin, g_cos)

        dist_to_goal = np.hypot(gx - x, gy - y)
        heading_err  = (goal_heading - heading + np.pi) % (2 * np.pi) - np.pi

        if dist_to_goal < CLOSE:
            # ── Phase 2: heading-alignment controller ──────────────────────
            steer = float(np.clip(3.0 * heading_err / np.pi, -1.0, 1.0))
            if abs(heading_err) < 0.35 and dist_to_goal > 0.006:
                accel = float(min(0.4, dist_to_goal * 6.0))
            elif dist_to_goal <= 0.006:
                accel = -0.3   # gentle brake once very close
            else:
                accel = 0.0    # wait for heading to align
        else:
            # ── Phase 1: grid policy navigation (γ-dependent quality) ─────
            r, c  = xy_to_rc(x, y)
            s     = max(0, min(N_STATES - 1, rc_to_s(r, c)))
            dr, dc = DELTAS[int(pi[s])]
            tr    = max(0, min(GRID_ROWS - 1, r + dr))
            tc    = max(0, min(GRID_COLS - 1, c + dc))
            tx, ty = rc_to_xy(tr, tc)

            desired = np.arctan2(ty - y, tx - x)
            diff    = (desired - heading + np.pi) % (2 * np.pi) - np.pi
            steer   = float(np.clip(4.0 * diff / np.pi, -1.0, 1.0))
            accel   = float(min(0.55, dist_to_goal * 4.0))

        obs, reward, done, truncated, _ = env.step(
            np.array([steer, accel], dtype=np.float32)
        )
        total_reward += reward
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        if done or truncated:
            break

    return frames, total_reward, V, pi, n_iter, goal_rc, start_rc


def frames_to_gif(frames, fps: int = 10) -> bytes:
    if not frames:
        return b""
    pil_frames = [Image.fromarray(f.astype(np.uint8)) for f in frames]
    buf = io.BytesIO()
    pil_frames[0].save(buf, format="GIF", save_all=True,
                       append_images=pil_frames[1:], loop=0,
                       duration=int(1000 / fps), optimize=False)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
#  Value-function visualisation  (grid heatmap, educational)
# ══════════════════════════════════════════════════════════════════════════════

def plot_value_heatmap(V, goal_rc, start_rc, pi=None) -> plt.Figure:
    """Show V(s) as a 2-D colour grid aligned with the normalised coordinate space."""
    V_grid = V.reshape(GRID_ROWS, GRID_COLS)

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(_DARK)
    ax.set_facecolor(_DARK)

    im = ax.imshow(V_grid, cmap=_CMAP, aspect="auto",
                   extent=[X_MIN, X_MAX, Y_MAX, Y_MIN])   # y-axis: top=Y_MIN
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="V(s)")

    # Goal marker
    gx, gy = rc_to_xy(*goal_rc)
    ax.plot(gx, gy, "*", color="white", markersize=14, zorder=5,
            label=f"goal ({goal_rc[0]},{goal_rc[1]})")

    # Start marker
    sx, sy = rc_to_xy(*start_rc)
    ax.plot(sx, sy, "o", color="#f59e0b", markersize=9, zorder=5,
            label=f"start ({start_rc[0]},{start_rc[1]})")

    # Policy arrows (subsampled)
    if pi is not None:
        arrow = {0: (0, -CELL_H*0.4), 1: (0, CELL_H*0.4),
                 2: (-CELL_W*0.4, 0), 3: (CELL_W*0.4, 0)}
        for s in range(N_STATES):
            r, c = s_to_rc(s)
            cx, cy = rc_to_xy(r, c)
            ddx, ddy = arrow[int(pi[s])]
            ax.annotate("", xy=(cx+ddx, cy+ddy), xytext=(cx, cy),
                        arrowprops=dict(arrowstyle="->", color="#94a3b8",
                                        lw=0.6, mutation_scale=7))

    ax.set_xlabel("normalised x  (actual x / 100 m)", color="#94a3b8", fontsize=8)
    ax.set_ylabel("normalised y",                      color="#94a3b8", fontsize=8)
    ax.set_title("State-Value Function V(s) — parking grid",
                 color="#e2e8f0", fontsize=9, pad=6)
    ax.tick_params(colors="#64748b", labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor("#1e293b")
    ax.legend(loc="upper right", fontsize=7,
              facecolor=_DARK, edgecolor="#1e293b", labelcolor="#94a3b8")
    fig.tight_layout()
    return fig


def plot_gamma_sweep(results) -> plt.Figure:
    n   = len(results)
    fig, axes = plt.subplots(1, n, figsize=(2.8*n, 3.5))
    fig.patch.set_facecolor(_DARK)
    vmin = min(r["V"].min() for r in results)
    vmax = max(r["V"].max() for r in results)
    for ax, r in zip(axes, results):
        ax.set_facecolor(_DARK)
        ax.imshow(r["V"].reshape(GRID_ROWS, GRID_COLS), cmap=_CMAP,
                  aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(f"γ={r['gamma']}", color="#e2e8f0", fontsize=8, pad=4)
        ax.axis("off")
        ax.set_xlabel(f"iters: {r['iters']}", color="#64748b", fontsize=7)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  Streamlit UI
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="app-title">🅿️ MDP γ Explorer — Parking</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">'
    'Gymnasium parking-v0 · 2-D Grid MDP · Value Iteration · '
    'Continuous Action Rollout'
    '</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 🎛️ Parameters")
    gamma = st.slider("Discount factor γ", 0.00, 0.999, 0.90, 0.01,
                      format="%.3f")
    horizon_str = f"≈ {1/(1-gamma):.1f} steps" if gamma < 0.999 else "∞"
    st.caption(f"**Effective horizon:** {horizon_str}")

    st.divider()
    st.markdown("### ⚙️ Value Iteration")
    theta    = st.select_slider("Convergence θ",
                                options=[1e-2, 1e-3, 1e-4, 1e-5], value=1e-4,
                                format_func=lambda x: f"{x:.0e}")
    max_iter = st.number_input("Max iterations", 50, 5000, 1000, 50)
    ep_steps = st.slider("Rollout steps", 50, 400, 250)
    fps      = st.slider("GIF fps", 5, 20, 10)

    st.divider()
    run_btn   = st.button("▶  Run", type="primary", use_container_width=True)
    sweep_btn = st.button("🔄  Compare γ ∈ {0, 0.3, 0.5, 0.7, 0.9, 0.99}",
                          use_container_width=True)

with st.expander("📖 How the grid MDP maps to parking-v0", expanded=False):
    st.markdown(r"""
**parking-v0** uses `KinematicsGoal` observation:
`[x, y, vx, vy, cos θ, sin θ]` normalised by scales `[100, 100, 5, 5, 1, 1]`.

We overlay a $20 \times 10$ grid on the normalised space
$x \in [-0.25, 0.25],\; y \in [-0.12, 0.12]$ (≈ 50 m × 24 m).

$$V^*(r,c) = \max_a \bigl[R(r,c,a) + \gamma \, V^*(T(r,c,a))\bigr]$$

At every rollout step the agent's real normalised $(x,y)$ is mapped to a
grid cell, $\pi^*(s)$ gives a direction (UP/DOWN/LEFT/RIGHT), and a heading
controller converts that to a continuous `[steering, acceleration]` command.

**γ effect:** low γ → flat value function → no directional gradient → the
agent wanders.  High γ → the goal cell's $+10$ reward propagates across
the whole grid → clear gradient → agent navigates to the parking spot.
    """)


# ── Main run ───────────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("Importing highway-env …"):
        gymnasium = import_highway()

    status = st.status("Running …", expanded=True)
    with status:
        st.write("🏗️  Creating `parking-v0` …")
        env = make_env(gymnasium)

        st.write(f"⚙️  Building grid MDP + Value Iteration  (γ={gamma:.3f}) …")
        t0 = time.time()
        frames, total_reward, V, pi, n_iter, goal_rc, start_rc = run_episode(
            env, gamma, float(theta), int(max_iter), n_steps=ep_steps
        )
        env.close()
        elapsed = time.time() - t0
        st.write(f"    → VI converged in **{n_iter}** iterations · "
                 f"{len(frames)} frames · reward = {total_reward:.3f} "
                 f"({elapsed:.1f}s)")
        status.update(label="✅  Done!", state="complete")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("γ (gamma)",         f"{gamma:.3f}")
    c2.metric("VI iterations",     n_iter)
    c3.metric("Episode reward",    f"{total_reward:.3f}")
    c4.metric("Effective horizon", horizon_str)

    if gamma < 0.5:
        note = (f"At γ={gamma:.2f} the value function is nearly flat — "
                f"the +10 goal reward is discounted to "
                f"{gamma**int(GRID_ROWS*GRID_COLS/4):.3f} from the far side "
                f"of the grid.  The agent has almost no directional gradient "
                f"and is likely to wander or not reach the goal.")
    elif gamma < 0.8:
        note = (f"Horizon ≈ {horizon_str}.  The goal's value begins to "
                f"propagate across the grid, but distant cells still have "
                f"weak gradients.  The agent may find the goal if it starts "
                f"nearby, but struggles from far away.")
    else:
        note = (f"Horizon ≈ {horizon_str}.  VI needed **{n_iter}** iterations. "
                f"The value mountain peaks at the goal cell and the policy "
                f"points toward it from every reachable cell — the agent "
                f"navigates directly to the parking spot.")

    st.markdown(f'<div class="teach-box">{note}</div>', unsafe_allow_html=True)
    st.divider()

    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown('<div class="section-hdr">Value Function V(s) on grid</div>',
                    unsafe_allow_html=True)
        st.pyplot(plot_value_heatmap(V, goal_rc, start_rc, pi=pi),
                  use_container_width=True)
    with col_b:
        st.markdown('<div class="section-hdr">Episode Animation (parking-v0)</div>',
                    unsafe_allow_html=True)
        if frames:
            gif = frames_to_gif(frames, fps=fps)
            st.image(gif,
                     caption=f"γ={gamma:.3f} · {len(frames)} frames · "
                             f"reward={total_reward:.3f}",
                     use_container_width=True)
            with st.expander("🖼️  Frame scrubber"):
                idx = st.slider("Frame", 0, len(frames)-1, 0)
                st.image(frames[idx], caption=f"Frame {idx+1}/{len(frames)}")
        else:
            st.warning("No frames captured.")

    st.session_state["last"] = dict(gamma=gamma, V=V, pi=pi,
                                    goal_rc=goal_rc, start_rc=start_rc)


# ── Gamma sweep (value functions only — no rollout for speed) ─────────────────
if sweep_btn:
    SWEEP_GAMMAS = [0.0, 0.3, 0.5, 0.7, 0.9, 0.99]
    with st.spinner("Importing highway-env …"):
        gymnasium = import_highway()

    st.markdown("### 📊 γ Sweep — value function shape across discount factors")
    prog    = st.progress(0)
    results = []

    # Use a fixed goal_rc / start_rc for fair comparison
    env = make_env(gymnasium)
    obs0, _ = env.reset()
    env.close()
    ego0, goal0 = _extract_obs(obs0)
    fixed_start = xy_to_rc(float(ego0[0]),  float(ego0[1]))
    fixed_goal  = xy_to_rc(float(goal0[0]), float(goal0[1]))

    for k, g in enumerate(SWEEP_GAMMAS):
        prog.progress((k+1)/len(SWEEP_GAMMAS), text=f"VI  γ={g} …")
        mdp_s = build_mdp(fixed_goal, fixed_start)
        V_s, pi_s, it_s = value_iteration(mdp_s, g, theta=1e-4, max_iter=2000)
        results.append({"gamma": g, "V": V_s, "pi": pi_s, "iters": it_s})

    prog.empty()
    st.pyplot(plot_gamma_sweep(results), use_container_width=True)

    import pandas as pd
    st.dataframe(
        pd.DataFrame([{"γ": r["gamma"], "VI iterations": r["iters"],
                       "V(start)": f"{r['V'][rc_to_s(*fixed_start)]:.2f}",
                       "V(goal)":  f"{r['V'][rc_to_s(*fixed_goal)]:.2f}"}
                      for r in results]),
        use_container_width=True,
    )
    st.markdown(
        '<div class="teach-box">'
        'Notice V(start) at low γ — it barely differs from V elsewhere, '
        'so the agent has no gradient to follow.  '
        'As γ → 1 the value mountain grows taller and steeper, '
        'giving the heading controller a strong directional signal.'
        '</div>',
        unsafe_allow_html=True,
    )

if not run_btn and not sweep_btn and "last" not in st.session_state:
    st.info("👈  Set γ and click **▶ Run** to begin.", icon="🅿️")
    st.markdown("""
**What to observe:**
- **γ = 0.1** — flat value function, agent wanders, rarely finds the goal
- **γ = 0.7** — gradient appears, agent navigates if started nearby
- **γ = 0.95** — strong value mountain, direct path to goal
- **γ Sweep** — compare V(s) shapes side by side without running rollouts
    """)
