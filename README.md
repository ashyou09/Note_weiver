# 🎓 NotesMaster AI

<p align="center">
  <strong>AI-powered study notes generator — runs 100% locally, built on the claw-code harness infrastructure</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/model-Qwen3.5%20via%20Ollama-blueviolet?style=for-the-badge" alt="Model" />
  <img src="https://img.shields.io/badge/backend-Flask%20%2B%20Python-blue?style=for-the-badge&logo=python" alt="Backend" />
  <img src="https://img.shields.io/badge/infra-claw--code%20harness-orange?style=for-the-badge" alt="Infrastructure" />
  <img src="https://img.shields.io/badge/runs-100%25%20local-green?style=for-the-badge" alt="Local" />
  <img src="https://img.shields.io/badge/GitHub-instructkr%2Fclaw--code-black?style=for-the-badge&logo=github" alt="GitHub" />
</p>

---

## What is NotesMaster AI?

**NotesMaster AI** is a local, privacy-first study notes generator that turns any topic, pasted text, or uploaded PDF into beautifully structured HTML study notes — powered by the **Qwen3.5 model running locally via Ollama**.

The entire backend is wired through the **claw-code Python harness** (`src/`), which provides session management, prompt routing, transcript history, and workspace context — the same infrastructure that powers the Claw Code agent runtime, repurposed here as a notes-generation engine.

> **No cloud. No API keys. No data leaves your machine.**

---

## Features

| Feature | Details |
|---|---|
| 📝 **Topic Mode** | Type any subject — AI generates complete structured notes from scratch |
| 📋 **Paste Mode** | Paste a lecture transcript, article, or raw text — AI converts it into an organized study guide |
| 📁 **Upload Mode** | Drop a PDF, TXT, or MD file — AI extracts and structures the content |
| ⚡ **Live Streaming** | See the AI write notes token by token in real time |
| 🖥 **Preview Panel** | Rendered HTML preview, raw HTML view, and one-click download |
| 💾 **Session Memory** | Powered by claw's `QueryEnginePort` + `TranscriptStore` — context persists across requests |
| 🔄 **History Panel** | All generated notes saved locally; reload any previous note instantly |
| 🤖 **Model Selector** | Switch between Qwen3.5 variants (0.8b → 9b) per-request |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  NotesMaster AI                         │
│                                                         │
│   notesmaster/static/index.html  ──── Dark glassmorphism│
│          │  (Flask SSE streaming UI)                    │
│          ▼                                              │
│   notesmaster/server.py          ──── Flask backend     │
│          │                                              │
│          ├──► src/query_engine.py   (session routing)   │
│          ├──► src/transcript.py     (history compaction) │
│          ├──► src/session_store.py  (JSON persistence)  │
│          ├──► src/history.py        (event timeline)    │
│          ├──► src/runtime.py        (prompt routing)    │
│          └──► src/system_init.py    (workspace context) │
│                                                         │
│   Ollama ─── Qwen3.5 (OpenAI-compat /v1/chat/completions)│
└─────────────────────────────────────────────────────────┘
```

### claw-code Infrastructure Used

NotesMaster sits **on top of** the claw-code Python harness rather than being a standalone Flask app. Here's what each claw component does inside the notes pipeline:

| claw Module | Role in NotesMaster |
|---|---|
| `QueryEnginePort` | Session management — tracks turn count, budget tokens, compact threshold |
| `TranscriptStore` | Stores the last N message pairs so follow-up requests have context |
| `StoredSession` / `save_session` | Persists session state to `.sessions/*.json` between HTTP requests |
| `HistoryLog` | Per-request event timeline (routing → session → messages → done) |
| `PortRuntime` | Prompt router — scores topic/content against tool and command manifests |
| `build_system_init_message` | Injects workspace context into the system prompt |

---

## Repository Layout

```
notes_weiver/
├── notesmaster/
│   ├── server.py          # Flask backend — routes, SSE streaming, claw session wiring
│   ├── requirements.txt   # flask, requests, pypdf
│   ├── static/
│   │   └── index.html     # Dark glassmorphism UI (pure HTML/CSS/JS, no framework)
│   ├── notes/             # Generated HTML notes saved here
│   ├── uploads/           # Temp storage for uploaded files
│   └── .sessions/         # claw JSON session files
│
├── src/                   # claw-code Python harness (the infrastructure layer)
│   ├── query_engine.py    # QueryEnginePort — session + turn management
│   ├── transcript.py      # TranscriptStore — rolling context history
│   ├── session_store.py   # JSON session persistence
│   ├── history.py         # HistoryLog — per-request event log
│   ├── runtime.py         # PortRuntime — prompt routing + tool matching
│   ├── system_init.py     # Workspace context builder
│   ├── commands.py        # Command port metadata
│   ├── tools.py           # Tool port metadata
│   ├── models.py          # Dataclasses for subsystem/module state
│   ├── main.py            # CLI entrypoint for manifest & summary inspection
│   └── ...                # Full harness: plugins, skills, coordinator, etc.
│
├── rust/                  # Rust port of the claw-code CLI (standalone runtime)
│   └── crates/
│       ├── claw-cli/      # Interactive REPL + one-shot agent binary
│       ├── api-client/    # Provider abstraction, OAuth, streaming
│       ├── runtime/       # Session, MCP, prompt construction
│       ├── tools/         # Tool manifest + execution framework
│       ├── commands/      # Slash commands + skills discovery
│       └── plugins/       # Plugin model + hook pipeline
│
├── tests/                 # Python harness verification tests
├── assets/                # Screenshots and visual assets
└── README.md              # This file
```

---

## Quickstart

### Prerequisites

- **Python 3.11+**
- **[Ollama](https://ollama.com/)** installed and running locally
- The **Qwen3.5** model pulled:

```bash
ollama pull qwen3.5:4b
```

### Install & Run

```bash
# 1. Clone the repo
git clone https://github.com/instructkr/claw-code.git
cd claw-code

# 2. Install Python dependencies
pip install -r notesmaster/requirements.txt

# 3. Start Ollama (if not already running)
ollama serve

# 4. Launch NotesMaster
python notesmaster/server.py
```

Then open **http://localhost:8080** in your browser.

---

## Usage

### Mode 1 — Topic Mode
Type any subject into the input box and click **Generate Notes**.

```
Examples:
  • Python Decorators and Closures
  • How does TCP/IP work internally?
  • Gradient Descent — intuition to math
  • Transformer Attention Mechanism
  • SOLID Principles in OOP
```

### Mode 2 — Paste Mode
Switch to **Paste Content**, paste any raw text (lecture notes, article, transcript), and generate.

### Mode 3 — Upload Mode
Switch to **Upload PDF / File**, drop a `.pdf`, `.txt`, or `.md` file, then generate.

---

## Connecting to Hugging Face Spaces

You can deploy NotesMaster to a **Hugging Face Space** if you have a remote Ollama endpoint or want to use a Hugging Face Inference API model instead.

### Option A — Point to a Remote Ollama

Set environment variables in your Space settings:

```bash
OPENAI_BASE_URL=https://your-ollama-endpoint/v1
OPENAI_API_KEY=your-key-or-any-string
NOTES_MODEL=qwen3.5:4b
```

### Option B — Use Hugging Face Inference API (OpenAI-compat)

Hugging Face provides an OpenAI-compatible endpoint for hosted models:

```bash
OPENAI_BASE_URL=https://api-inference.huggingface.co/v1
OPENAI_API_KEY=hf_your_token_here
NOTES_MODEL=Qwen/Qwen2.5-7B-Instruct
```

### Space Configuration (`README.md` frontmatter for HF)

Add this YAML block at the top of your repo's README when pushing to a Hugging Face Space:

```yaml
---
title: NotesMaster AI
emoji: 🎓
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---
```

### Push to Hugging Face

```bash
# Add Hugging Face as a remote (replace with your Space URL)
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/notesmaster

# Push
git push hf main
```

---

## Note Output Format

Every generated note is a **self-contained HTML file** with:

- Google Fonts (Crimson Pro, Source Sans 3, JetBrains Mono)
- Deep navy color scheme with light-blue tinted code/formula boxes
- Orange/green/red left-border highlight cards
- Two-column comparison grids
- Step flow diagrams for algorithms
- Syntax-highlighted code blocks
- Key Takeaways section (8–12 points)

Notes are saved to `notesmaster/notes/` with the pattern `{topic}_{YYYYMMDD_HHMMSS}.html`.

---

## Model Selection

| Model | Speed | Quality | Use Case |
|---|---|---|---|
| `qwen3.5:0.8b` | ⚡⚡⚡ Fastest | Good | Quick drafts |
| `qwen3.5:2b` | ⚡⚡ Fast | Better | Most topics |
| `qwen3.5:4b` | ⚡ Balanced | Best local | **Recommended** |
| `qwen3.5:9b` | 🐢 Slower | Excellent | Deep/technical topics |

---

## Claw Harness CLI (Python)

The `src/` Python harness exposes inspection tools for the underlying claw infrastructure:

```bash
# Render porting summary
python3 -m src.main summary

# Print workspace manifest
python3 -m src.main manifest

# List subsystems
python3 -m src.main subsystems --limit 16

# Inspect tool shims
python3 -m src.main tools --limit 10

# Inspect command metadata
python3 -m src.main commands --limit 10

# Simulate prompt routing
python3 -m src.main route "gradient descent explanation"

# Run parity audit
python3 -m src.main parity-audit

# Run tests
python3 -m unittest discover -s tests -v
```

---

## Rust CLI (claw-code runtime)

The Rust workspace in `rust/` is an independent, production-grade agent CLI:

```bash
cd rust
cargo build --release

# Authenticate (for cloud providers)
cargo run --release -- login

# Interactive REPL
cargo run --release

# One-shot prompt
cargo run --release -- -p "What files are in this directory?"

# Use local Ollama model
export OPENAI_BASE_URL="http://127.0.0.1:11434/v1"
export OPENAI_API_KEY="ollama"
cargo run --release -- --model qwen3.5:4b
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_BASE_URL` | `http://127.0.0.1:11434/v1` | Ollama or any OpenAI-compat endpoint |
| `OPENAI_API_KEY` | `ollama` | API key (can be any string for local Ollama) |
| `NOTES_MODEL` | `qwen3.5:4b` | Default model for note generation |

---

## Tech Stack

- **Frontend**: Pure HTML/CSS/JS — dark glassmorphism UI, no framework
- **Backend**: Flask 3.x with SSE streaming
- **AI Runtime**: Ollama (local) via OpenAI-compat `/v1/chat/completions`
- **Infrastructure**: claw-code Python harness (`src/`) for session, routing, transcript
- **PDF Extraction**: `pypdf`
- **Rust CLI**: Tokio, Axum, serde — full async agent runtime

---

## GitHub → Hugging Face Sync Workflow

```bash
# After making changes locally:
git add .
git commit -m "feat: describe your change"

# Push to GitHub
git push origin main

# Push to Hugging Face Space
git push hf main
```

---

## License & Affiliation

- This repository does **not** claim ownership of the original Claw Code source material.
- This repository is **not affiliated with, endorsed by, or maintained by Anthropic**.
- NotesMaster AI is an independent application built using the open-source harness layer.
