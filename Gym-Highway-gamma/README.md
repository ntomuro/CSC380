# [Extra] Gymnasium γ (gamma) Explorer — Highway-env Streamlit Testbed App

Interactive demo for demonstrating the **discount factor γ** in Markov Decision Processes (MDP),
using ![Gymnasium](https://gymnasium.farama.org/)'s ![Highway-Env](https://highway-env.farama.org/quickstart/) environment with real Value Iteration.

## :rocket: Note

Three environments are implemented as testbeds:

- **Highway** — drive on a multilane highway populated with other vehicles
- **Merge** -- drive on a multilane highway near a road junction with incoming vehicles on the access ramp
- **Parking** -- park in a given space with the appropriate heading

![highway](https://raw.githubusercontent.com/eleurent/highway-env/gh-media/docs/media/highway.gif)
![merge](https://raw.githubusercontent.com/eleurent/highway-env/gh-media/docs/media/merge-env.gif)
![parking](https://raw.githubusercontent.com/eleurent/highway-env/gh-media/docs/media/parking-env.gif)

## :sheep: Objective

The objective is to see how much the discount factor γ influences the policy learning in different environments (i.e., how different environments are sensitive to γ).  Since γ control the future reward, a smaller γ negatively affect the learning for environments that are long-horizon, interaction-heavy tasks (which requires long-term planning), while the effect of γ is not so critical for environments that are short-horizon or dense-reward tasks (because feedback is immediate).

The three enviroments are selected to demonstrate:

- **Highway** — moderately sensitive because long horizon driving
- **Merge** -- highly sensitive because delayed cooperative behavior on the access ramp
- **Parking** -- less sensitive because Dense reward shaping (distance, angle, etc.)

More detailed explanations are available in the document **"More-explanations-on-Gymnasium γ.pdf"**.

---

## 🚀 Quick Start (change the script file to "highway_app.py", "merge_app.py", "parking_app.py" or "roundabout_app.py)

```bash
# 0. Create a virtual environment
python -m venv venv
source venv/bin/activate     # or venv\Scripts\activate for Windows PC

# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run highway_app.py
```

Your browser will open automatically at `http://localhost:8501`.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `gymnasium` | RL environment API |
| `highway-env` | Highway driving MDP |
| `finite-mdp` | Converts highway-env to discrete MDP |
| `numpy` | Value Iteration computation |
| `matplotlib` | V(s) and policy plots |
| `Pillow` | Converts frames to animated GIF |

> **Linux/Mac note:** If rendering fails, install a virtual display:
> ```bash
> sudo apt-get install xvfb
> pip install pyvirtualdisplay
> ```
> Then prefix your run command:
> ```bash
> xvfb-run -a streamlit run highway_app.py
> ```

---

## 🎛️ Features

- **γ number input + slider** — change and re-run instantly
- **Real Value Iteration** on `env.to_finite_mdp()` (TTC state space)
- **Animated GIF** of the actual highway-env episode render
- **Frame scrubber** — step through frames one by one
- **V(s) heatmap** — value function across all MDP states
- **Policy bar chart** — action distribution of the optimal policy
- **Teaching note** — auto-generated explanation tailored to each γ range
- **γ Sweep** — compares γ ∈ {0, 0.3, 0.5, 0.7, 0.9, 0.99} side by side

---

## 🎓 Classroom Use

1. Project the browser window
2. Start with γ=0.0 — show the myopic agent (just goes fast, ignores collisions)
3. Increase to γ=0.9 — show how the value function gains structure
4. Hit **γ Sweep** — compare all six γ values at once
5. Ask students: *"Why does VI need more iterations as γ→1?"*

---

*Prepared for AI Foundations — Markov Decision Processes module*
