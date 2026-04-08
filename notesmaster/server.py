"""
server.py — NotesMaster AI
---------------------------
Modes:
  MODE 1 — TOPIC:   user types a topic → AI generates structured notes from scratch
  MODE 2 — CONTENT: user pastes text OR uploads PDF/txt/md → AI makes notes FROM that content

AI Backend (auto-detected, in priority order):
  1. Local Ollama  (OPENAI_BASE_URL defaults to http://127.0.0.1:11434/v1)
  2. OpenRouter API (OPENROUTER_API_KEY env var)

Claw infrastructure used:
  - QueryEnginePort   → session management + turn tracking
  - TranscriptStore   → history with auto-compaction
  - StoredSession     → JSON session persistence
  - HistoryLog        → per-request event timeline
  - PortRuntime       → prompt routing
  - build_system_init_message() → workspace context
"""

import sys, os, json, time, io, subprocess, shutil
from pathlib import Path
from datetime import datetime
from flask import Flask, request, Response, send_from_directory, jsonify, stream_with_context
import requests as http

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

# ── repo root → claw src/ ──────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.query_engine import QueryEnginePort, QueryEngineConfig
from src.session_store import StoredSession, save_session, load_session
from src.transcript import TranscriptStore
from src.history import HistoryLog
from src.runtime import PortRuntime
from src.system_init import build_system_init_message

# ── config ─────────────────────────────────────────────────────────────────
OLLAMA_URL    = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1")
OLLAMA_KEY    = os.environ.get("OPENAI_API_KEY",  "ollama")
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "qwen/qwen-plus")
DEFAULT_MODEL = os.environ.get("NOTES_MODEL", "qwen3.5:2b")
PORT          = int(os.environ.get("PORT", 7860))

SESSION_DIR   = Path(__file__).parent / ".sessions"
NOTES_DIR     = Path(__file__).parent / "notes"
UPLOAD_DIR    = Path(__file__).parent / "uploads"
for d in [SESSION_DIR, NOTES_DIR, UPLOAD_DIR]:
    d.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB max upload


# ═══════════════════════════════════════════════════════════════════════════
# BACKEND DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def ollama_available() -> bool:
    """Check if local Ollama is running."""
    try:
        r = http.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False

def get_backend() -> str:
    """Return primary backend depending on what's available."""
    if ollama_available():
        return "ollama"
    return "openrouter" # Always fallback to Public OpenRouter Inference


# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_BASE = """\
I am giving you [a YouTube video transcript / lecture notes / a document / code files] about [topic name]. Your job is to produce a single, complete, self-contained HTML file of deep study notes. The notes must be detailed enough that someone can learn the entire topic from just these notes — without ever needing to watch the video or read the original source.

SECTION A — LENGTH RULES (Most Important, Non-Negotiable)
These rules apply to every single section without exception:

Every concept gets 2–4 sentences of explanation minimum — never a one-liner
Every formula must be written out → each variable explained in plain language → followed by a fully worked numerical example with real numbers
Every example from the source must be completely worked out step by step — not just mentioned or referenced
Every function or algorithm must be explained as: what goes in → what happens inside step by step → what comes out
Every code snippet must be explained: plain-English description first → full syntax-highlighted code block → line-by-line output explanation
Every comparison (A vs B) must explain both sides fully — never just label them
Every pro or con must get 2–3 full sentences — never just a word or a label
Do NOT summarize — if the source spent time on something, your notes spend time on it too
Do NOT pad — say each thing once, say it well, move on
Target length: A 30-minute video should produce notes long enough to replace watching it entirely
SECTION B — CONTENT REQUIREMENTS
Follow these rules for every piece of content in the source:

Opening of every section:
Start by answering "what is this concept and why does it matter in the real world or in this subject?" — before going into any details. This gives the reader orientation before depth.

Definitions:
Write the formal definition first, then restate it in simple plain language (and in hinglish if it helps it click). Never just write a dictionary definition and move on.

Formulas:

Write the formula in a formula box using JetBrains Mono font
Below it, explain every single variable/symbol in plain English — what it represents, what unit it has, what range it can take
Then show a fully worked numerical example: plug in real numbers, show each step, show the final result
If the formula has edge cases or special conditions, mention them
Code (Critical — Read Carefully):
Code must never appear as random lines dumped together. Every code block must follow this sequence:

Plain English description — what does this function/block do, why is it written, what problem does it solve
The full code block — properly indented, properly sequenced (imports first → class/function definitions → main logic → output calls), syntax highlighted with the exact colors defined in Section D
Line-by-line or block-by-block explanation — after the code block, explain what each logical group of lines does in sequence
Output box — show what the output looks like when the code runs, in a green result box
Code must be written in logical sequence, never randomly ordered:

text

ORDER RULE FOR CODE:
  Step 1 → Imports and library loading
  Step 2 → Constants, config, hyperparameters
  Step 3 → Data loading or input definition
  Step 4 → Data preprocessing / transformation
  Step 5 → Model definition / core logic / function definitions
  Step 6 → Training loop / main computation
  Step 7 → Evaluation / testing
  Step 8 → Output / print / visualization
If a code example from the source is out of this order, reorder it logically while keeping all the original lines intact.

Comparisons (A vs B):
Always use a two-column card layout. Each side must have: definition of what it is, when to use it, its strengths, its weaknesses. Never just put two labels in two columns.

Algorithms and Processes:
Always use a step-flow layout: numbered dot → connecting line → card with title and explanation. Each step card must explain what happens in that step and why, not just name it.

Flow Diagrams (for important processes):
When a concept involves a sequence of decisions or data transformations (like a training pipeline, a sorting algorithm, a model architecture), represent it as an HTML/CSS flow diagram using boxes and arrows made from divs and unicode arrow characters (→, ↓, ⬇). Do not use external images. The flow should read top-to-bottom or left-to-right and each box should have a label and a one-line description inside it. Use this for things like: neural network forward pass, data preprocessing pipeline, algorithm decision tree, HTTP request lifecycle, etc.

Advantages vs Disadvantages:
Always place them side by side — green card on left, red card on right. Each point inside must have 2–3 sentences of real explanation, not just a word label.

Hinglish:
Add hinglish naturally only when it makes a concept click faster. Never force it. Wrap it in a styled hinglish tag. Example: "Yeh basically ek shortcut hai jo model ko overfit hone se rokta hai — jaise exam mein sirf ek hi type ke questions practice karo toh real exam mein fail ho jaoge."

Tables:
Use tables for structured comparisons of 3+ items. Dark navy header row, alternating light blue rows, all text in deep navy. Never use a table for just 2 items — use cards for that.

Key Takeaways (Last Section Always):
Write 8 to 12 numbered points. Each point must be at least 2 full sentences — one sentence stating the fact, one sentence explaining why it matters or how to use it. Never write one-line takeaways.

SECTION C — WHAT TO AVOID (Strictly)
❌ Do NOT write one-line bullet points for complex ideas
❌ Do NOT skip any example that appears in the source content
❌ Do NOT merge two different concepts into one vague paragraph
❌ Do NOT add filler phrases ("As we can see...", "It is worth noting...", "In conclusion...")
❌ Do NOT repeat the same point twice in different words
❌ Do NOT make a section that is just a heading followed by 3 short bullets
❌ Do NOT use dark backgrounds on any element — everything is light
❌ Do NOT dump code randomly — always follow the logical sequence rule above
❌ Do NOT write code explanations after the entire file — explain each block right where it appears
❌ Do NOT use external images, CDN icons, or SVG files — all visuals must be pure HTML/CSS
SECTION D — EXACT TEXT AND COLOR RULES
These colors are mandatory. The AI must not guess or substitute.

General Text Colors:
Element	Color	Hex
Body text (p, li)	Deep navy	#1a1a2e
H1 and H2 headings	Very dark navy	#0d1b4b
H3 subheadings	Dark slate blue	#1e3a5f
H4 card headings	Dark blue	#0d3d8c
Section number labels	Muted grey	#9aa5b4
Small uppercase labels	Medium blue	#3d5c99
Italic notes / footnotes	Medium blue	#3d5c99
<strong> inside paragraphs	Dark navy	#0d1b4b
Formula Box:
Element	Color	Hex
Box background	Light blue tint	#f4f7fe
Box border	Soft blue	#b8ccf0
Formula text (math)	Dark navy	#0d1b4b
Formula title label	Medium blue	#3d5c99
Explanation lines (// comments)	Steel blue	#5c7ab0
Code Block — Syntax Colors (ALL code must use these exactly):
Token Type	Color	Hex	Style
Code block background	Light blue tint	#f4f7fe	—
Default code text	Dark navy	#0d1b4b	normal
Keywords (def, for, if, import, return, class, while, in, and, or, not, True, False, None)	Dark blue	#0d3d8c	bold
Function names (after def or when called)	Dark purple	#7a1fa2	bold
Strings (anything in quotes)	Dark red-brown	#8b2500	normal
Comments (# lines)	Dark green	#1a5c2a	italic
Numbers and numeric literals	Dark teal	#0d5c3a	bold
Variable names	Dark blue	#0055aa	normal
Imported class names / library names	Dark purple	#7a1fa2	bold
Code block label/title	Medium blue	#3d5c99	—
Inline code (backtick style)	—	bg #e8eef8, text #0d3d8c	—
Important: Code must be written as HTML <span> tags with style="color: ..." applied to each token type. Do not use any external syntax highlighting library. Write the color spans manually for every keyword, function name, string, comment, number, and variable in the code block.

Highlight / Callout Boxes:
Box Type	Background	Left Border	Text	Strong Text
Insight / Key point (orange)	#fffbf0	#c87d00	#1a1a2e	#7a4400
Good / Advantage (green)	#f0faf4	#1a7a44	#1a1a2e	#0d4a27
Warning / Disadvantage (red)	#fff5f5	#b91c1c	#1a1a2e	#7f1d1d
Definition box	#eef4ff	— border #93b8ef	label #0d3d8c	—
Output / Result box	#f0faf4	— border #7abf96	#0d3d1e	label #0d5c2a
Table Colors:
Element	Color
Header background	#0d3d8c
Header text	#ffffff
Even rows	#f4f7fe
Odd rows	#ffffff
Cell text	#1a1a2e
Cell border	#d0daea
Card Colors:
Card Type	Background	Border	Heading Color
Regular info card	#f9fbff	#b8ccf0	#0d3d8c
Advantage card	#f0faf4	#1a7a44	#0d4a27
Disadvantage card	#fff5f5	#b91c1c	#7f1d1d
Flow diagram box	#eef4ff	#93b8ef	#0d1b4b
Hinglish Tag:
Property	Value
Background	#fff5e6
Border	2px solid #e8a838
Text color	#7a3800
Font style	italic
Font	Source Sans 3
Step Flow (Algorithm Steps):
Element	Value
Dot color	#1a6ef5
Connecting line	#c8d8f5
Card background	#f4f7fe
Card border	#c8d8f5
Card strong text	#0d3d8c
Dividers:
<hr> between every major section: border: none; border-top: 1.5px solid #c8d8f5;
SECTION E — LAYOUT AND DESIGN RULES
Fonts (load from Google Fonts):

text

https://fonts.googleapis.com/css2?family=Crimson+Pro:wght@400;600;700&family=Source+Sans+3:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap
H1, H2: Crimson Pro, serif
All body text, paragraphs, list items: Source Sans 3, sans-serif
All code, formulas, labels, section numbers: JetBrains Mono, monospace
Background: #ffffff everywhere — absolutely no dark backgrounds on any element, ever.

H2 headings: Always have border-left: 4px solid #1a6ef5; padding-left: 12px;

Section numbers: Small JetBrains Mono text in #9aa5b4 written above every H2 as 01, 02, 03...

Comparisons: Always CSS Grid, grid-template-columns: 1fr 1fr, gap 20px.

Algorithms: Always step-flow layout — vertical stack of: [dot] ── [card with step title and explanation], dot is a filled circle #1a6ef5, connected by a vertical line #c8d8f5.

Flow Diagrams: Build using nested div elements. Each box is a styled div. Arrows between boxes use → or ↓ in styled spans. No images, no SVGs, no external libraries. The diagram must be readable and not overflow on normal screen widths.

Advantages vs Disadvantages: Always side-by-side, green card left, red card right, inside a CSS grid.

Print styles: Include @media print { body { padding: 20px; } .no-print { display: none; } } at the bottom of the <style> block.

SECTION F — DOCUMENT STRUCTURE (Follow This Exactly)
text

<html>
  <head>
    → Google Fonts link
    → Full <style> block with all CSS rules
  </head>
  <body>
    → HEADER BLOCK
        - Topic name in large Crimson Pro, color #0d1b4b
        - Subtitle in Source Sans 3, color #2d3a5c
        - Subject/series label in small caps JetBrains Mono, color #9aa5b4

    → SECTION 01 — Prerequisites / Background Context
        (What prior knowledge is needed? What is the context for this topic?)

    → SECTION 02 — First major concept from the source
    → SECTION 03 — Second major concept from the source
    → ... (one section per major concept, in the same order as the source)

    → SECOND-TO-LAST SECTION — Code Implementation
        (Only if code appears in the source — follow all code rules from Section B)

    → LAST SECTION — Key Takeaways
        (8–12 numbered points, 2 full sentences each minimum)

    → HR dividers between every section
  </body>
</html>
SECTION G — FLOW DIAGRAM CONSTRUCTION RULE
When you need to show a process, pipeline, or decision sequence, build it like this in HTML/CSS:

text

[Box: Step 1 Name]
    Short description inside the box
        ↓
[Box: Step 2 Name]
    Short description inside the box
        ↓
[Box: Step 3 Name — Decision]
    Yes → [Box A]    No → [Box B]
Each box is a div with:

background: #eef4ff
border: 1px solid #93b8ef
border-radius: 8px
padding: 12px 18px
max-width: 420px
font: Source Sans 3, color #1a1a2e
Title inside in <strong> color #0d1b4b
Arrows are <div style="text-align:center; font-size: 1.5rem; color: #1a6ef5;">↓</div> between boxes.

For branching decisions, use a CSS flex row with two boxes side by side and label them "Yes →" and "No →".

"""

def build_topic_prompt(topic: str) -> str:
    return (
        f"Create complete, professional study notes for this topic:\n\n"
        f"TOPIC: {topic}\n\n"
        f"Follow the structured teaching flow strictly. Generate a FULL HTML document."
    )

def build_content_prompt(content: str, source_name: str) -> str:
    return (
        f"The user has provided content from '{source_name}'. "
        f"Study this content carefully and create COMPLETE, STRUCTURED notes from it.\n\n"
        f"SOURCE: {source_name}\n"
        f"CONTENT ({len(content)} characters):\n"
        f"---\n{content}\n---\n\n"
        f"Extract ALL key concepts, organize them using the structured teaching flow, "
        f"and generate a FULL HTML study guide document from this material."
    )


# ═══════════════════════════════════════════════════════════════════════════
# PDF EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def extract_pdf_text(file_bytes: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text.strip())
    return "\n\n".join(pages)


# ═══════════════════════════════════════════════════════════════════════════
# STREAMING — Ollama (OpenAI-compat)
# ═══════════════════════════════════════════════════════════════════════════

def ollama_stream(messages: list[dict], model: str):
    url = f"{OLLAMA_URL}/chat/completions"
    payload = {"model": model, "messages": messages, "stream": True,
               "max_tokens": 8192, "temperature": 0.7, "top_p": 0.9,
               "frequency_penalty": 0.5, "presence_penalty": 0.3}
    headers = {"Authorization": f"Bearer {OLLAMA_KEY}", "Content-Type": "application/json"}
    try:
        with http.post(url, json=payload, headers=headers, stream=True, timeout=(10, 600)) as resp:
            if resp.status_code != 200:
                yield f"[ERROR {resp.status_code}]: {resp.text[:300]}"
                return
            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw.decode() if isinstance(raw, bytes) else raw
                if line.startswith("data: "):
                    line = line[6:]
                if line.strip() == "[DONE]":
                    break
                try:
                    delta = json.loads(line)["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield {"type": "content", "text": delta}
                except Exception:
                    continue
    except Exception as e:
        yield {"type": "content", "text": f"\n\n[LOCAL OLLAMA ERROR]: The local AI model timed out or crashed {str(e)[:100]}\n\n"}


# ═══════════════════════════════════════════════════════════════════════════
# STREAMING — OpenRouter
# ═══════════════════════════════════════════════════════════════════════════

def openrouter_stream(messages: list[dict], model: str):
    """Stream from OpenRouter Using OpenAI client."""
    if not HAS_OPENAI:
        yield {"type": "content", "text": "[ERROR] The openai python package is required."}
        return

    or_model = model if ("/" in model and model != "qwen-plus") else OPENROUTER_MODEL
    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_KEY,
        default_headers={
            "HTTP-Referer": "https://github.com/ashyou09/Note_weiver",
            "X-OpenRouter-Title": "NotesMaster AI",
        }
    )

    try:
        resp = client.chat.completions.create(
            model=or_model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            temperature=0.7,
            max_tokens=8192,
            extra_body={"reasoning": {"enabled": True}}
        )
        for chunk in resp:
            # Handle usage chunk (usually final)
            if hasattr(chunk, 'usage') and chunk.usage:
                try:
                    # Capture reasoning tokens if present (OpenRouter extension)
                    # It might be in extra_fields or a specific sub-object
                    r_tokens = getattr(chunk.usage, 'reasoning_tokens', 0)
                    if r_tokens:
                        yield {"type": "meta", "reasoning_tokens": r_tokens}
                except:
                    pass

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            reasoning = getattr(delta, "reasoning", None) # Some models stream reasoning here

            if content:
                yield {"type": "content", "text": content}
            if reasoning:
                yield {"type": "reasoning", "text": reasoning}
    except Exception as e:
        yield {"type": "content", "text": f"\n\n[OPENROUTER ERROR]: The model failed to respond {str(e)[:100]}\n\n"}


def auto_stream(messages: list[dict], model: str, backend: str):
    """Route to correct backend logic."""
    if "/" in model or backend == "openrouter" or "claude" in model or "nemotron" in model:
        yield from openrouter_stream(messages, model)
    else:
        yield from ollama_stream(messages, model)


# ═══════════════════════════════════════════════════════════════════════════
# CLAW SESSION
# ═══════════════════════════════════════════════════════════════════════════

def get_engine(session_id: str | None) -> QueryEnginePort:
    if session_id:
        try:
            e = QueryEnginePort.from_saved_session(session_id)
            e.config = QueryEngineConfig(max_turns=30, max_budget_tokens=200_000, compact_after_turns=20)
            return e
        except Exception:
            pass
    e = QueryEnginePort.from_workspace()
    e.config = QueryEngineConfig(max_turns=30, max_budget_tokens=200_000, compact_after_turns=20)
    return e


# ═══════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/status")
def status():
    backend = get_backend()
    if backend == "ollama":
        try:
            r = http.get("http://localhost:11434/api/tags", timeout=3)
            models = [m["name"] for m in r.json().get("models", [])]
            return jsonify({"ok": True, "ollama": True, "openrouter": True, "models": models})
        except Exception as e:
            return jsonify({"ok": False, "ollama": False, "openrouter": True, "error": str(e)})
    else:
        return jsonify({"ok": True, "ollama": False, "openrouter": True, "model": OPENROUTER_MODEL})


@app.route("/api/upload", methods=["POST"])
def upload():
    """Extract text from uploaded file (PDF, txt, md)."""
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "No file"}), 400
    fname = f.filename or "upload"
    data = f.read()
    ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
    try:
        if ext == "pdf":
            text = extract_pdf_text(data)
        else:
            text = data.decode("utf-8", errors="replace")
        text = text.strip()
        if not text:
            return jsonify({"error": "Could not extract text from file"}), 400
        return jsonify({"text": text, "chars": len(text), "name": fname})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/generate", methods=["POST"])
def generate():
    """
    SSE streaming generation.
    body: { mode: 'topic'|'content', topic?: str, content?: str,
            source_name?: str, model: str, session_id?: str }
    """
    d = request.get_json()
    mode       = d.get("mode", "topic")
    topic      = (d.get("topic") or "").strip()
    content    = (d.get("content") or "").strip()
    src_name   = (d.get("source_name") or "pasted content").strip()
    model      = d.get("model") or DEFAULT_MODEL
    session_id = d.get("session_id") or None

    if mode == "topic" and not topic:
        return jsonify({"error": "No topic provided"}), 400
    if mode == "content" and not content:
        return jsonify({"error": "No content provided"}), 400

    backend = get_backend()
    if backend == "none":
        return jsonify({"error": "No AI backend available. Start Ollama or set OPENROUTER_API_KEY."}), 503

    def stream():
        nonlocal model
        # ── claw infrastructure ──────────────────────────────────────────
        history = HistoryLog()
        runtime = PortRuntime()
        query   = topic if mode == "topic" else src_name
        matches = runtime.route_prompt(query, limit=5)
        history.add("routing", f"mode={mode} matches={len(matches)} backend={backend}")

        engine = get_engine(session_id)
        history.add("session", f"id={engine.session_id} turns={len(engine.mutable_messages)}")

        workspace_ctx = build_system_init_message(trusted=True)
        full_system = SYSTEM_BASE + f"\n\n<!-- Workspace: {workspace_ctx} -->"

        # For small local models, rigorous complex instructions cause repetition. Use simplified prompt.
        if "2b" in model.lower() or "3b" in model.lower() or "4b" in model.lower():
            simple_sys = (
                "You are NotesMaster AI. You MUST generate study notes about the requested topic.\n"
                "Constraints:\n"
                "1. Always output ONLY valid HTML wrapped inside a single <div>.\n"
                "2. Use <h1>, <h2>, <p>, <ul>, <li> tags strictly for structure.\n"
                "3. Stop generating once the summary is complete. Do not repeat sections."
            )
            messages = [{"role": "system", "content": simple_sys}]
        else:
            messages = [{"role": "system", "content": full_system}]
            
        for m in engine.transcript_store.replay()[-10:]:
            if m.startswith("U:"):
                messages.append({"role": "user", "content": m[2:].strip()})
            elif m.startswith("A_JSON:"):
                try:
                    data = json.loads(m[7:].strip())
                    msgs = {"role": "assistant", "content": data.get("content", "")}
                    if data.get("reasoning_details"):
                        msgs["reasoning_details"] = data["reasoning_details"]
                    messages.append(msgs)
                except:
                    messages.append({"role": "assistant", "content": m[7:].strip()})
            elif m.startswith("A:"):
                messages.append({"role": "assistant", "content": m[2:].strip()})

        user_msg = build_topic_prompt(topic) if mode == "topic" else build_content_prompt(content, src_name)
        messages.append({"role": "user", "content": user_msg})
        history.add("messages", f"len={len(messages)} mode={mode} backend={backend}")

        # ── metadata event ───────────────────────────────────────────────
        yield f"data: {json.dumps({'type':'meta','session_id':engine.session_id,'model':model,'mode':mode,'backend':backend,'routes':[{'kind':m.kind,'name':m.name,'score':m.score} for m in matches[:4]]})}\n\n"

        # ── stream tokens ────────────────────────────────────────────────
        start  = time.time()
        content_chunks = []
        reasoning_chunks = []
        for ev in auto_stream(messages, model, backend):
            if isinstance(ev, str):
                text = ev
                content_chunks.append(text)
                yield f"data: {json.dumps({'type':'chunk', 'text': text})}\n\n"
            else:
                etype = ev.get("type", "content")
                if etype == "content":
                    text = ev.get("text", "")
                    content_chunks.append(text)
                    yield f"data: {json.dumps({'type':'chunk', 'text': text})}\n\n"
                elif etype == "reasoning":
                    text = ev.get("text", "")
                    reasoning_chunks.append(text)
                    yield f"data: {json.dumps({'type':'reasoning', 'text': text})}\n\n"
                elif etype == "meta":
                    # Pass through metadata like reasoning_tokens
                    yield f"data: {json.dumps(ev)}\n\n"

        elapsed  = round(time.time() - start, 1)
        raw_html = "".join(content_chunks).strip()
        full_reasoning = "".join(reasoning_chunks)

        # strip markdown fences if model wrapped output
        if raw_html.startswith("```html"):
            raw_html = raw_html[7:]
        elif raw_html.startswith("```"):
            raw_html = raw_html[3:]
        if raw_html.endswith("```"):
            raw_html = raw_html[:-3]
        raw_html = raw_html.strip()

        # ── claw session update ──────────────────────────────────────────
        try:
            engine.submit_message(
                f"Notes: {topic or src_name}",
                matched_commands=tuple(m.name for m in matches if m.kind == "command"),
                matched_tools=tuple(m.name for m in matches if m.kind == "tool"),
            )
            engine.transcript_store.append(f"U: {user_msg[:500]}")
            if full_reasoning:
                store_data = {"content": raw_html, "reasoning_details": full_reasoning}
                engine.transcript_store.append(f"A_JSON:{json.dumps(store_data)}")
            else:
                engine.transcript_store.append(f"A: {raw_html}")
            engine.compact_messages_if_needed()
            session_path = engine.persist_session()
            history.add("done", f"elapsed={elapsed}s session={session_path}")
        except Exception as e:
            print("Claw session update failed:", e)

        # ── save HTML ────────────────────────────────────────────────────
        try:
            safe = "".join(c for c in (topic or src_name)[:40] if c.isalnum() or c in " -_").strip().replace(" ", "_")
            if not safe:
                safe = "notes"
            fname = f"{safe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            (NOTES_DIR / fname).write_text(raw_html, encoding="utf-8")
        except Exception as e:
            fname = f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            print("Failed to save HTML tightly:", e)
            (NOTES_DIR / fname).write_text(raw_html, encoding="utf-8")

        yield f"data: {json.dumps({'type':'done','elapsed':elapsed,'note_file':fname,'session_id':engine.session_id})}\n\n"

    return Response(
        stream_with_context(stream()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/notes", methods=["GET"])
def list_notes():
    notes = []
    for f in sorted(NOTES_DIR.glob("*.html"), reverse=True)[:30]:
        notes.append({
            "name": f.name,
            "size": f.stat().st_size,
            "ts": datetime.fromtimestamp(f.stat().st_mtime).strftime("%d %b %H:%M"),
        })
    return jsonify({"notes": notes})


@app.route("/api/notes", methods=["DELETE"])
def delete_all_notes():
    """Delete every saved note HTML file."""
    deleted = 0
    for f in NOTES_DIR.glob("*.html"):
        try:
            f.unlink()
            deleted += 1
        except Exception:
            pass
    return jsonify({"deleted": deleted})


@app.route("/api/notes/<filename>", methods=["GET"])
def get_note(filename):
    return send_from_directory(str(NOTES_DIR), filename)


@app.route("/api/notes/<filename>", methods=["DELETE"])
def delete_note(filename):
    """Delete a single saved note by filename."""
    # Sanitise: only allow plain filenames (no path traversal)
    safe = Path(filename).name
    target = NOTES_DIR / safe
    if not target.exists():
        return jsonify({"error": "Not found"}), 404
    try:
        target.unlink()
        return jsonify({"deleted": safe})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/sessions")
def list_sessions():
    sessions = []
    for f in sorted(SESSION_DIR.glob("*.json"), reverse=True)[:15]:
        try:
            d = json.loads(f.read_text())
            sessions.append({
                "id": d.get("session_id", f.stem),
                "messages": len(d.get("messages", [])),
                "tokens_in": d.get("input_tokens", 0),
                "tokens_out": d.get("output_tokens", 0),
            })
        except Exception:
            pass
    return jsonify({"sessions": sessions})


@app.route("/api/start-local", methods=["POST"])
def start_local():
    """Attempt to verify and start Ollama dynamically if requested."""
    data = request.get_json() or {}
    model = data.get("model", "qwen3.5:4b")
    
    if not shutil.which("ollama"):
        return jsonify({"ok": False, "error": "Ollama executable not found on this system. You must download it manually."})
    
    try:
        # Check if already running first
        r = http.get("http://localhost:11434/api/tags", timeout=1)
        if r.status_code == 200:
            return jsonify({"ok": True, "message": "Ollama is ready and running!"})
    except Exception as e:
        return jsonify({"ok": False, "error": f"Ollama daemon does not appear to be running on port 11434. Error: {e}"})

# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    backend = get_backend()
    print(f"\n{'═'*54}")
    print(f"  🎓  NotesMaster AI")
    print(f"  🤖  Model : {DEFAULT_MODEL}")
    print(f"  🔌  Backend: {backend.upper()} {'('+OPENROUTER_MODEL+')' if backend=='openrouter' else ''}")
    print(f"  🌐  http://localhost:{PORT}")
    print(f"{'═'*54}\n")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
