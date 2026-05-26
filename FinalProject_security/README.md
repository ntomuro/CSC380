# Final Project: Multi-Agent Adversarial Prompt Injection Detection System

## Overview

In this project you will build an agentic AI system that analyzes LLM conversations and
determines whether they contain adversarial prompt injection attempts. The system uses
**LangGraph** to orchestrate three specialized agents that run in parallel and synthesize
their findings into a final security verdict.

This is **defensive AI security** — you are building a detection system, not an attack tool.

Assignment page: https://condor.depaul.edu/ntomuro/courses/380/2026spring/assign/FinalProject/finalproj-2026spring.html

## Setup

### 1. Create a virtual environment

```
python -m venv venv
```

### 2. Activate it

- Windows: `venv\Scripts\activate`
- Mac/Linux: `source venv/bin/activate`

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Set your OpenAI API key

Copy `.env.example` to `.env` and fill in your key:

```
cp .env.example .env
# then edit .env and replace sk-... with your real key
```

### 5. Launch the notebook

```
jupyter notebook starter/prompt_injection_detection.ipynb <== to be updated (!)
```

## Project Structure

```
FinalProject_security/
├── requirements.txt
├── .env.example
├── README.md
├── data/
│   └── synthetic_dataset.py    <- 60 labeled conversations
├── agents/
│   ├── state.py                <- LangGraph state schema (you fill this in)
│   ├── intent_analysis_agent.py
│   ├── instruction_hierarchy_agent.py
│   └── risk_classification_agent.py
├── graph/
│   └── detection_graph.py      <- LangGraph graph (you wire the edges)
├── evaluation/
│   └── metrics.py              <- evaluation utilities
└── starter/
    └── prompt_injection_detection.ipynb  <- your working notebook
```

## Submission

Submit your completed notebook as a `.ipynb` file with **all cell outputs present**.

Required outputs:
- The Mermaid diagram in Cell 11 must be rendered (proves correct graph topology)
- The confusion matrix plot in Cell 15 must be visible
- All evaluation metric values must be printed

## Academic Integrity

You may use LLM assistants to understand concepts, but all code you submit must be your own.
The TODOs in the notebook represent the core intellectual contribution of this project.
