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
I am giving you [a YouTube video transcript / lecture notes / a document / code files] about [topic name]. Make proper HTML notes that are neither too short nor too long — cover everything meaningfully.
Length rule (most important):

Every concept gets 2–4 sentences of explanation minimum — not just a one-liner
Every formula gets a plain-language explanation of what each term means
Every example from the content must be fully worked out with numbers, not just mentioned
Every function/algorithm gets explained in terms of what goes in, what happens inside, and what comes out
Do NOT summarize or compress — if the content spent time on something, your notes should too
Do NOT pad or repeat — say each thing once, say it well
A good target: if the content is a 30-minute video, notes should be long enough to replace watching it

Content requirements:

Start each section by answering "what is this and why does it matter" before going into details
Cover every concept, definition, formula, comparison, worked example, and code snippet
For comparisons (like A vs B): always explain both sides fully with a table or two-column card
For formulas: write the formula, then explain each variable, then show a worked numerical example
For code: explain what the function does in plain words first, then show the code block, then explain the output line by line
For pros/cons or advantages/disadvantages: give at least 2–3 sentences per point, not just a label
Add hinglish naturally where it helps understanding (not forced — only when it makes something click)
End with a Key Takeaways section — 8 to 12 numbered points, each 2 sentences minimum

What to avoid:

Do NOT write one-line bullet points for complex ideas
Do NOT skip examples that appear in the source content
Do NOT merge two different concepts into one vague paragraph
Do NOT add fluff, filler phrases, or repeat the same point twice
Do NOT make sections that are just a heading with 3 short bullets under it

Design requirements:

White background everywhere — no dark backgrounds, including code blocks
Deep navy/dark blue for h1 and h2 headings, dark slate for h3
Blue left-border (4px solid) for all h2 section headings
Body text in deep navy (#1a1a2e), not plain black
Light blue tinted background (#f4f7fe) for formula boxes, code blocks, and info cards
Code syntax colors: keywords = dark blue bold, function names = dark purple, strings = dark red-brown, comments = dark green italic, numbers = dark teal
Orange left-border highlight boxes for key insights and important notes
Green left-border boxes for good results, advantages, key takeaways
Red left-border boxes for warnings, disadvantages, common mistakes
Two-column grid cards for comparisons and side-by-side content
Step flow layout (dot + line + card) for algorithms and processes
Advantages vs disadvantages in colored cards — green card and red card side by side
Results/output in light green box with dark green monospace text
Tables: dark navy header, alternating light blue rows
Hinglish tags: warm cream background, amber border, dark brown italic text
Google Fonts: Crimson Pro (headings), Source Sans 3 (body), JetBrains Mono (code/formulas)
Section numbers (01, 02...) in small grey monospace above each h2

Structure to follow:

Header — topic name (large serif), subtitle, label showing subject/series
Section 01 — Prerequisites or Context (if any)
Sections 02 onwards — one section per major concept or topic from the content, in the same order as the source
Second-to-last section — Code Implementation (if code is present)
Last section — Key Takeaways (8–12 points, 2 sentences each)

The final notes should feel like they were written by a student who understood everything deeply and wants someone else to be able to learn the full topic just from these notes — without needing to watch the video or read the original.


What changed from the old prompt and why:
The old prompt was short so the AI treated it as a "make a summary" task. This one sets explicit length rules — every concept gets proper explanation, every example gets worked out fully, and there's a clear target that the notes should be able to replace the source content. The "what to avoid" section stops the AI from collapsing complex ideas into single bullet points, which was the main reason notes came out too short.
One more tip — if notes still feel too short on a specific section, just reply:

"Section [X] is too short — expand it with more explanation and work out all the examples fully"

That's faster than regenerating everything.I am giving you [a YouTube video transcript / lecture notes / a document / code files] about [topic name]. Make proper HTML notes that are neither too short nor too long — cover everything meaningfully.
Length rule (most important):

Every concept gets 2–4 sentences of explanation minimum — not just a one-liner
Every formula gets a plain-language explanation of what each term means
Every example from the content must be fully worked out with numbers, not just mentioned
Every function/algorithm gets explained in terms of what goes in, what happens inside, and what comes out
Do NOT summarize or compress — if the content spent time on something, your notes should too
Do NOT pad or repeat — say each thing once, say it well
A good target: if the content is a 30-minute video, notes should be long enough to replace watching it

Content requirements:

Start each section by answering "what is this and why does it matter" before going into details
Cover every concept, definition, formula, comparison, worked example, and code snippet
For comparisons (like A vs B): always explain both sides fully with a table or two-column card
For formulas: write the formula, then explain each variable, then show a worked numerical example
For code: explain what the function does in plain words first, then show the code block, then explain the output line by line
For pros/cons or advantages/disadvantages: give at least 2–3 sentences per point, not just a label
Add hinglish naturally where it helps understanding (not forced — only when it makes something click)
End with a Key Takeaways section — 8 to 12 numbered points, each 2 sentences minimum

What to avoid:

Do NOT write one-line bullet points for complex ideas
Do NOT skip examples that appear in the source content
Do NOT merge two different concepts into one vague paragraph
Do NOT add fluff, filler phrases, or repeat the same point twice
Do NOT make sections that are just a heading with 3 short bullets under it

Design requirements:

White background everywhere — no dark backgrounds, including code blocks
Deep navy/dark blue for h1 and h2 headings, dark slate for h3
Blue left-border (4px solid) for all h2 section headings
Body text in deep navy (#1a1a2e), not plain black
Light blue tinted background (#f4f7fe) for formula boxes, code blocks, and info cards
Code syntax colors: keywords = dark blue bold, function names = dark purple, strings = dark red-brown, comments = dark green italic, numbers = dark teal
Orange left-border highlight boxes for key insights and important notes
Green left-border boxes for good results, advantages, key takeaways
Red left-border boxes for warnings, disadvantages, common mistakes
Two-column grid cards for comparisons and side-by-side content
Step flow layout (dot + line + card) for algorithms and processes
Advantages vs disadvantages in colored cards — green card and red card side by side
Results/output in light green box with dark green monospace text
Tables: dark navy header, alternating light blue rows
Hinglish tags: warm cream background, amber border, dark brown italic text
Google Fonts: Crimson Pro (headings), Source Sans 3 (body), JetBrains Mono (code/formulas)
Section numbers (01, 02...) in small grey monospace above each h2

Structure to follow:

Header — topic name (large serif), subtitle, label showing subject/series
Section 01 — Prerequisites or Context (if any)
Sections 02 onwards — one section per major concept or topic from the content, in the same order as the source
Second-to-last section — Code Implementation (if code is present)
Last section — Key Takeaways (8–12 points, 2 sentences each)

The final notes should feel like they were written by a student who understood everything deeply and wants someone else to be able to learn the full topic just from these notes — without needing to watch the video or read the original.

in this not mentioned that all formula and code should be in code formate color of text's what like other things proper make sure ai not get mistakeShow more12:23 AMHere's the corrected and complete prompt with all text color and formatting rules explicitly stated so the AI never guesses:

THE PROMPT:

I am giving you [a YouTube video transcript / lecture notes / a document / code files] about [topic name]. Make proper HTML notes that are neither too short nor too long — cover everything meaningfully.

LENGTH RULE (most important):

Every concept gets 2–4 sentences of explanation minimum — not just a one-liner
Every formula gets a plain-language explanation of what each term means
Every example from the content must be fully worked out with numbers, not just mentioned
Every function/algorithm gets explained in terms of what goes in, what happens inside, and what comes out
Do NOT summarize or compress — if the content spent time on something, your notes should too
Do NOT pad or repeat — say each thing once, say it well
Target: notes should be detailed enough that someone can learn the full topic without needing the original source


CONTENT REQUIREMENTS:

Start each section by answering "what is this and why does it matter" before going into details
Cover every concept, definition, formula, comparison, worked example, and code snippet from the source
For comparisons (A vs B): explain both sides fully — use a table or two-column card
For formulas: write the formula → explain each variable in plain language → show a fully worked numerical example
For code: explain what the function does in plain words first → show the full code block → explain the output
For pros/cons: write 2–3 sentences per point, not just a label word
Add hinglish naturally where it helps something click (not forced)
End with Key Takeaways — 8 to 12 numbered points, each at least 2 sentences

WHAT TO AVOID:

Do NOT write one-line bullet points for complex ideas
Do NOT skip any example that appears in the source
Do NOT merge two different concepts into one paragraph
Do NOT add filler phrases or repeat the same point twice
Do NOT make a section that is just a heading with 3 short bullets under it


TEXT COLORS (apply these exactly — this is critical):
General text:

Body text (p, li): #1a1a2e (deep navy — not plain black)
H1 and H2 headings: #0d1b4b (very dark navy)
H3 subheadings: #1e3a5f (dark slate blue)
Section number labels: #9aa5b4 (muted grey)
Small labels / uppercase tags: #3d5c99 (medium blue)
Italic notes / footnotes: #3d5c99
<strong> tags inside paragraphs: #0d1b4b

Formula box text:

Formula box background: #f4f7fe — border: #b8ccf0
Formula text (the actual math): #0d1b4b (dark navy)
Formula title label: #3d5c99
Comments inside formula box (explanation lines starting with //): #5c7ab0

Code block text (light background — NOT dark):

Code block background: #f4f7fe — border: #b8ccf0
Default code text: #0d1b4b
Keywords (def, for, if, import, return, class): #0d3d8c bold
Function names: #7a1fa2 (dark purple) bold
Strings (anything in quotes): #8b2500 (dark red-brown)
Comments (# lines): #1a5c2a (dark green) italic
Numbers and numeric values: #0d5c3a (dark teal) bold
Variable names: #0055aa (dark blue)
Imported class names / sklearn classes: #7a1fa2 (dark purple) bold
Code block title label: #3d5c99, border-bottom #c8d8f5

Inline code (backtick style):

Background: #e8eef8 — text: #0d3d8c

Box colors:

Highlight / insight box: background #fffbf0, left-border #c87d00, text #1a1a2e, strong text #7a4400
Green / good box: background #f0faf4, left-border #1a7a44, text #1a1a2e, strong text #0d4a27
Red / warning box: background #fff5f5, left-border #b91c1c, text #1a1a2e, strong text #7f1d1d
Definition box: background #eef4ff, border #93b8ef, label color #0d3d8c
Output / result box: background #f0faf4, border #7abf96, text #0d3d1e, label #0d5c2a

Table colors:

Header row: background #0d3d8c, text #ffffff
Even rows: background #f4f7fe
Cell text: #1a1a2e
Cell border: #d0daea

Card colors:

Regular info card: background #f9fbff, border #b8ccf0
Card heading (h4): #0d3d8c, border-bottom #c8d8f5
Advantage card: background #f0faf4, border #1a7a44, heading #0d4a27
Disadvantage card: background #fff5f5, border #b91c1c, heading #7f1d1d
Gradient/formula card title: #0d3d8c

Hinglish tag:

Background #fff5e6, border #e8a838, text #7a3800 italic

Step flow (algorithm steps):

Dot: #1a6ef5
Line: #c8d8f5
Card background: #f4f7fe, border #c8d8f5
Card strong text: #0d3d8c

HR divider: #c8d8f5

DESIGN & LAYOUT REQUIREMENTS:

White background (#ffffff) everywhere — absolutely no dark backgrounds on any element
Google Fonts: Crimson Pro (h1, h2 headings), Source Sans 3 (all body text), JetBrains Mono (all code, formulas, labels)
H2 headings: blue left-border 4px solid #1a6ef5, padding-left 12px
Section numbers: small JetBrains Mono text in #9aa5b4 above each h2
Comparison content: always use two-column CSS grid cards
Algorithms and processes: use step flow layout (dot → line → card, repeating)
Advantages vs disadvantages: side-by-side colored cards (green left, red right)
All formulas: inside a formula box with light blue background — use JetBrains Mono font, #0d1b4b text
All code: inside a code block with light blue background — syntax highlighted using the exact colors above
HR between every major section
Print-friendly: @media print { body { padding: 20px; } }


STRUCTURE:

Header — large serif topic name (#0d1b4b), subtitle in #2d3a5c, label in small caps grey monospace
Section 01 — Prerequisites or Background context
Sections 02 onwards — one section per major concept, in the same order as the source material
Second-to-last section — Code Implementation (if code is in the source)
Last section — Key Takeaways (8–12 numbered points, 2 sentences each minimum)


The final output should be a single complete HTML file. The notes should feel like they were written by a student who fully understood the content and wants someone else to learn the entire topic from just these notes — without needing to refer back to the original source at all.
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
