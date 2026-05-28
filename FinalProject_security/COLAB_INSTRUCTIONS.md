# Running the Final Project on Google Colab

This guide walks you through opening, running, and saving the project notebooks on
Google Colab. You do **not** need to install anything on your own computer.

---

## What you need before you start

| Requirement | Notes |
|-------------|-------|
| Google account | Needed to use Colab and Google Drive |
| OpenAI API key | Get one at [platform.openai.com](https://platform.openai.com) — see Section 4 below |

---

## Step 1 — Open the scaffold notebook (do this first)

The scaffold notebook teaches you the LangGraph concepts you need before working on the
project. Read and run it before opening the project notebook.

**Click the link below to open it directly in Colab:**

[Open langgraph\_scaffold\_colab.ipynb in Colab](https://colab.research.google.com/github/ntomuro/CSC380/blob/main/starter/langgraph_scaffold_colab.ipynb)

> **Save a copy to your Drive immediately** (next step), otherwise your progress
> disappears when the Colab session ends.

### Save your copy

1. In the Colab menu bar, click **File → Save a copy in Drive**.
2. A new tab opens with a copy named `Copy of langgraph_scaffold_colab.ipynb` in
   your `My Drive/Colab Notebooks/` folder.
3. Rename it if you like: **File → Rename**.
4. Work in this copy from now on — it auto-saves to Drive.

---

## Step 2 — Open the project notebook

This is the file you will submit. It contains all the TODOs you need to complete.

**Click the link below to open it directly in Colab:**

[Open prompt\_injection\_detection\_colab.ipynb in Colab](https://colab.research.google.com/github/ntomuro/CSC380/blob/main/starter/prompt_injection_detection_colab.ipynb)

**Save a copy to Drive immediately** (same steps as above):
**File → Save a copy in Drive**

> Work only in your saved copy. Any changes made to the GitHub-linked version
> are not saved.

---

## Step 3 — Add your OpenAI API key (Colab Secrets)

The project calls the OpenAI API. The safest way to provide your key is through
**Colab Secrets**, which stores it once and never shows it in your notebook.

### Add the secret (one-time setup)

1. In the left sidebar, click the **key icon** (🔑).
2. Click **Add new secret**.
3. Set **Name** to exactly: `OPENAI_API_KEY`
4. Paste your key (starts with `sk-`) into the **Value** field.
5. Toggle **Notebook access** to ON.
6. Click **Save**.

![Colab Secrets panel](https://storage.googleapis.com/colab-cdn-v4-us/uploads/VPmpYmFnZWQ=)

> Your key is stored in your Google account and reused automatically every time
> you open Colab. You only need to do this once per Google account.

### If you skip Secrets

When you run the API key cell, a text box will appear asking you to paste your key.
Type or paste it and press Enter. The key is used for that session only and is not
stored anywhere.

---

## Step 4 — Run the setup cells

Open your saved copy of `prompt_injection_detection_colab.ipynb` and run the first
three cells in order:

| Cell | What it does | Time |
|------|-------------|------|
| **Cell 4** — Install packages | `!pip install langgraph openai ...` | ~30 seconds |
| **Cell 5** — Write helper modules | Writes the dataset and evaluation code to the Colab filesystem | < 1 second |
| **Cell 9** — API key | Loads your key from Secrets (or prompts you to paste it) | instant |

> **You must re-run these three cells every time you reconnect to a new runtime.**
> Colab runtimes are temporary — installed packages and written files are lost
> when the session disconnects. Your notebook edits are always safe in Drive.

---

## Step 5 — Complete the TODOs and run the notebook

Work through the notebook from top to bottom. Each TODO cell has clear instructions.
Run cells in order; do not skip ahead.

**Before submitting, verify all required outputs are present:**

- [ ] Cell 28 shows the Mermaid diagram with `__start__` having **two** outgoing arrows
- [ ] Cells 31–32 (smoke test) show all three agents' outputs for a single conversation
- [ ] Cell 34 ran for all 60 conversations (outputs visible)
- [ ] Cell 36 prints accuracy and F1 scores
- [ ] Cell 38 shows the confusion matrix plot
- [ ] Cells 40–42 have your error analysis and reflection answers written in

---

## Step 6 — Download and submit

1. In Colab: **File → Download → Download .ipynb**
2. The file saves to your computer's Downloads folder.
3. Rename it: `Lastname_Firstname_FinalProject.ipynb`
4. Upload to D2L.

> Make sure all cell outputs are visible in the downloaded file. If a cell shows
> no output, re-run it and download again.

---

## Alternative: clone the repo to Google Drive (persistent setup)

If you want a persistent workspace where installed packages survive disconnects,
mount your Drive and clone the repo there. Run these cells at the top of any Colab
notebook:

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
```

```python
# Clone the repo into your Drive (run once)
import os
REPO_DIR = '/content/drive/MyDrive/CSC380'
if not os.path.exists(REPO_DIR):
    !git clone https://github.com/ntomuro/CSC380 "{REPO_DIR}"
    print("Repo cloned.")
else:
    print("Repo already exists. Pulling latest changes...")
    !git -C "{REPO_DIR}" pull
```

```python
# Install packages into the Drive-based venv (persists across sessions)
!pip install -q -r "/content/drive/MyDrive/CSC380/requirements.txt"
```

After cloning, open your working notebook from Drive:

1. In Colab: **File → Open notebook → Google Drive**
2. Navigate to `My Drive / CSC380 / starter /`
3. Open `prompt_injection_detection_colab.ipynb`

> With this setup, you still need to re-run `!pip install` and the helper-files cell
> each session, but your notebook edits are automatically saved to Drive.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'langgraph'` | Re-run Cell 4 (pip install) |
| `ModuleNotFoundError: No module named 'data'` | Re-run Cell 5 (helper modules) |
| `OPENAI_API_KEY not set` | Re-run Cell 9 (API key) |
| `AuthenticationError` from OpenAI | Your API key is wrong or expired — check [platform.openai.com](https://platform.openai.com) |
| Outputs disappeared after disconnect | Re-run all cells from top; your code edits are still saved in Drive |
| "Runtime disconnected" during the 60-conversation loop | Re-run the eval loop cell; it will re-process all conversations |

---

*Questions? Post to the course discussion board or email the instructor at tomuro@cs.depaul.edu.*
