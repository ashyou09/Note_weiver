
# 🎓 NotesMaster AI

<p align="center">
  <strong>AI-powered study notes generator — local Ollama + Qwen3 cloud, built on the claw-code harness</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/model-Qwen3%20%7C%20Qwen3.5-blueviolet?style=for-the-badge" alt="Model" />
  <img src="https://img.shields.io/badge/backend-Flask%20%2B%20Python-blue?style=for-the-badge&logo=python" alt="Backend" />
  <img src="https://img.shields.io/badge/infra-claw--code%20harness-orange?style=for-the-badge" alt="Infrastructure" />
  <img src="https://img.shields.io/badge/HF%20Space-ashyou09%2FNotes__weiver-yellow?style=for-the-badge&logo=huggingface" alt="HF Space" />
  <img src="https://img.shields.io/badge/GitHub-ashyou09%2FNote__weiver-black?style=for-the-badge&logo=github" alt="GitHub" />
</p>

---

## Quick Start

```bash
# Clone from Hugging Face Space
git clone https://huggingface.co/spaces/ashyou09/Notes_weiver

# Or clone from GitHub
git clone https://github.com/ashyou09/Note_weiver.git
```

---

## What is NotesMaster AI?

**NotesMaster AI** is a privacy-first study notes generator that turns any topic, pasted text, or uploaded PDF into beautifully structured HTML study notes.

**Two AI backends supported — automatically detected:**
- 🖥️ **Local** — Qwen3.5 (any size) via Ollama running on your machine
- ☁️ **Cloud** — Qwen3-8B (latest) via Hugging Face Inference API (set `HF_TOKEN`)

> **No data leaves your machine in local mode.** Cloud mode sends content to HF Inference API.

---

## Features

| Feature | Details |
|---|---|
| 📝 **Topic Mode** | Type any subject — AI generates complete structured notes from scratch |
| 📋 **Paste Mode** | Paste lecture notes, articles, transcripts — AI organizes them into a study guide |
| 📁 **Upload Mode** | Drop a PDF, TXT, or MD file — AI extracts and structures the content |
| ⚡ **Live Streaming** | Watch AI write notes token-by-token in real time |
| 🖥 **Preview Panel** | Rendered HTML preview, raw HTML view, one-click download |
| 🆕 **Qwen3-8B Cloud** | Latest Qwen3 model via HF Inference API — no Ollama required |
| 🗑 **Delete Notes** | Remove individual notes or clear all saved notes with one click |
| 💾 **Session Memory** | Context persists across requests via claw's `QueryEnginePort` + `TranscriptStore` |
| 🔄 **History Panel** | All generated notes saved locally; reload any previous note instantly |
| 🤖 **Model Selector** | Switch between Qwen3.5 size variants or Qwen3-8B cloud per-request |

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
│   ┌──────────────────────────────────────────────────┐  │
│   │  AI Backend (auto-detected)                      │  │
│   │  1. Local Ollama → Qwen3.5:4b (default)          │  │
│   │  2. HF Inference API → Qwen3-8B (HF_TOKEN)       │  │
│   └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## Repository Layout

```
notes_weiver/
├── notesmaster/
│   ├── server.py          # Flask backend — routes, SSE streaming, dual-backend AI
│   ├── requirements.txt   # flask, requests, pypdf
│   ├── static/
│   │   └── index.html     # Dark glassmorphism UI — no framework needed
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
│   └── ...                # Full harness: plugins, skills, coordinator, etc.
│
├── rust/                  # Rust port of the claw-code CLI (standalone runtime)
├── tests/                 # Python harness verification tests
├── Dockerfile             # HF Spaces deployment (python:3.11-slim, port 7860)
└── README.md              # This file
```

---

## Quickstart — Local

### Prerequisites
- **Python 3.11+**
- **[Ollama](https://ollama.com/)** installed and running locally
- Qwen3.5 model pulled:

```bash
ollama pull qwen3.5:4b
```

### Run

```bash
# Clone
git clone https://github.com/ashyou09/Note_weiver.git
cd Note_weiver

# Install deps
pip install -r notesmaster/requirements.txt

# Start Ollama
ollama serve

# Launch NotesMaster
python notesmaster/server.py
```

Open **http://localhost:7860** in your browser.

---

## Quickstart — Hugging Face Space (Cloud)

```bash
git clone https://huggingface.co/spaces/ashyou09/Notes_weiver
cd Notes_weiver
```

The Space uses **Docker SDK** and runs on port `7860`. Set these secrets in your Space settings:

| Secret | Value | Required |
|---|---|---|
| `HF_TOKEN` | Your HF token (read access) | ✅ For Qwen3 cloud |
| `HF_MODEL` | `Qwen/Qwen3-8B` | Optional (this is the default) |
| `NOTES_MODEL` | `Qwen/Qwen3-8B` | Optional override |

Then the **Qwen3-8B ☁️** option in the model selector will stream from HF Inference API automatically.

---

## Usage

### Mode 1 — Topic Mode
Type any subject and click **Generate Notes** (or press `Ctrl+Enter`):

```
Examples:
  • Python Decorators and Closures
  • How does TCP/IP work internally?
  • Gradient Descent — intuition to math
  • Transformer Attention Mechanism
  • SOLID Principles in OOP
```

### Mode 2 — Paste Mode
Switch to **Paste Content**, paste raw text (lecture notes, articles, transcripts), generate.

### Mode 3 — Upload Mode
Switch to **Upload PDF / File**, drop a `.pdf`, `.txt`, or `.md` file, generate.

### Deleting Notes
- **Single note**: hover over any note in the Recent Notes panel → click the 🗑 trash icon
- **All notes**: click the **🗑 Clear All** button in the Recent Notes header

---

## Model Options

| Model | Type | Speed | Best For |
|---|---|---|---|
| `qwen3.5:0.8b` | Local | ⚡⚡⚡ | Quick drafts |
| `qwen3.5:2b` | Local | ⚡⚡ | Most topics |
| `qwen3.5:4b` | Local | ⚡ Balanced | **Default — recommended** |
| `qwen3.5:9b` | Local | 🐢 | Deep/technical topics |
| `Qwen3-8B` | ☁️ HF Cloud | ⚡ | Latest Qwen3, no Ollama needed |
| `Qwen2.5-7B` | ☁️ HF Cloud | ⚡ | Stable fallback cloud model |

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/status` | `GET` | Backend status (ollama / hf / none) |
| `/api/generate` | `POST` | SSE stream — generate notes |
| `/api/upload` | `POST` | Upload PDF/TXT/MD, returns extracted text |
| `/api/notes` | `GET` | List all saved notes |
| `/api/notes` | `DELETE` | Delete all saved notes |
| `/api/notes/<filename>` | `GET` | Get a specific note |
| `/api/notes/<filename>` | `DELETE` | Delete a specific note |
| `/api/sessions` | `GET` | List claw session files |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `7860` | Server port (7860 for HF Spaces) |
| `OPENAI_BASE_URL` | `http://127.0.0.1:11434/v1` | Local Ollama endpoint |
| `OPENAI_API_KEY` | `ollama` | API key (any string for local Ollama) |
| `HF_TOKEN` | _(empty)_ | HF token for Qwen3 cloud inference |
| `HF_MODEL` | `Qwen/Qwen3-8B` | HF model to use when `HF_TOKEN` is set |
| `NOTES_MODEL` | `qwen3.5:4b` | Default model shown in UI |

---

## Tech Stack

- **Frontend**: Pure HTML/CSS/JS — dark glassmorphism, animated, no framework
- **Backend**: Flask 3.x with SSE streaming
- **Local AI**: Ollama (Qwen3.5 family) via OpenAI-compat `/v1/chat/completions`
- **Cloud AI**: HF Inference API (Qwen3-8B) via OpenAI-compat endpoint
- **Infrastructure**: claw-code Python harness (`src/`) — session, routing, transcript
- **PDF Extraction**: `pypdf`
- **Deployment**: Docker on Hugging Face Spaces (port 7860)

---

## Deploy / Sync Workflow

```bash
# Push updates to GitHub
git add .
git commit -m "feat: your change"
git push origin main

# Push to Hugging Face Space
git push hf main
```

Remotes:
```bash
git remote add origin https://github.com/ashyou09/Note_weiver.git
git remote add hf     https://huggingface.co/spaces/ashyou09/Notes_weiver
```

---

## Note Output Format

Every generated note is a **self-contained HTML file** with:

- Google Fonts (Crimson Pro, Source Sans 3, JetBrains Mono)
- Deep navy color scheme with light-blue tinted code/formula boxes
- Orange/green/red left-border highlight cards
- Two-column comparison grids and step flow diagrams
- Syntax-highlighted code blocks
- Key Takeaways section (8–12 points)

Notes are saved to `notesmaster/notes/` as `{topic}_{YYYYMMDD_HHMMSS}.html`.
