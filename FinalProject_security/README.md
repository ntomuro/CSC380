# :japanese_goblin: CSC 380 Spring 2026 Final Project: 
# Multi-Agent Adversarial Prompt Injection Detection System

## Overview

In this project you will build an agentic AI system that analyzes LLM conversations and
determines whether they contain adversarial prompt injection attempts. The system uses
**LangGraph** to orchestrate three specialized agents that run in parallel and synthesize
their findings into a final security verdict.

This is **defensive AI security** — you are building a detection system, not an attack tool.

---

## Quick Start (Google Colab)

This project runs entirely on Google Colab. No local installation is required.

### Step 1 — Read the scaffold notebook first

The scaffold teaches you the LangGraph concepts you need before starting the project.

[Open langgraph_scaffold_colab.ipynb in Colab](https://colab.research.google.com/github/ntomuro/CSC380/blob/main/starter/langgraph_scaffold_colab.ipynb)

### Step 2 — Open the project notebook

[Open prompt_injection_detection_colab.ipynb in Colab](https://colab.research.google.com/github/ntomuro/CSC380/blob/main/starter/prompt_injection_detection_colab.ipynb)

Save a copy to your Drive immediately: **File → Save a copy in Drive**

### Step 3 — Follow COLAB_INSTRUCTIONS.md

See **[COLAB_INSTRUCTIONS.md](COLAB_INSTRUCTIONS.md)** for the complete setup walkthrough:
API key configuration, which cells to run first, and troubleshooting.

---

## Repository Structure

```
FinalProject_security/
+-- ASSIGNMENT-ALL.md           <- full code specification
+-- COLAB_INSTRUCTIONS.md       <- step-by-step Colab setup guide for students
+-- README.md
+-- requirements.txt            <- for local development only
+-- .env
+-- .env.example                <- template for local API key
+-- data/
|   +-- synthetic_dataset.py    <- 60 labeled conversations (embedded in Colab notebook)
+-- agents/                     <- reference implementations (do not copy)
|   +-- state.py
|   +-- intent_analysis_agent.py
|   +-- instruction_hierarchy_agent.py
|   +-- risk_classification_agent.py
+-- graph/
|   +-- detection_graph.py      <- reference LangGraph graph (do not copy)
+-- evaluation/
|   +-- metrics.py              <- evaluation utilities (embedded in Colab notebook)
+-- starter/
|   +-- prompt_injection_detection_colab.ipynb   <- **STUDENT WORKING FILE (Colab)**
|   +-- prompt_injection_detection.ipynb         <- local version (optional)
|   +-- langgraph_scaffold_colab.ipynb           <- LangGraph tutorial (read first)
```

---

## Local Development (Optional)

If you prefer to run the notebook locally instead of on Colab:

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your OpenAI API key
cp .env.example .env
# Edit .env and replace sk-... with your real key

# 4. Launch the notebook
jupyter notebook starter/prompt_injection_detection.ipynb
```

---

## Assignment Details

- [Code Assignment Specifications](ASSIGNMENT-ALL.md)
- **[Overall Assignment Page](https://condor.depaul.edu/ntomuro/courses/380/2026spring/assign/FinalProject/finalproj-2026spring.html)** - This is THE assignment page.

---

## Submission

1. In Colab: **File → Download → Download .ipynb**
2. Rename: `Lastname_Firstname_FinalProject.ipynb`
3. Upload to D2L with **all cell outputs present**.

---

## Academic Integrity

You may use LLM assistants to understand concepts.  **HOWEVER, all code you submit must be your own.**
The TODOs in the notebook represent the core intellectual contribution of this project.
