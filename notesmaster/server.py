"""
server.py — NotesMaster AI
---------------------------
Modes:
  MODE 1 — TOPIC:   user types a topic → AI generates structured notes from scratch
  MODE 2 — CONTENT: user pastes text OR uploads PDF/txt/md → AI makes notes FROM that content

AI Backend (auto-detected, in priority order):
  1. Local Ollama  (OPENAI_BASE_URL defaults to http://127.0.0.1:11434/v1)
  2. Hugging Face Inference API (HF_TOKEN env var)

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
HF_TOKEN      = os.environ.get("HF_TOKEN", "")
HF_MODEL      = os.environ.get("HF_MODEL", "Qwen/Qwen3-8B")
DASHSCOPE_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
DEFAULT_MODEL = os.environ.get("NOTES_MODEL", "qwen3.5:4b")
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
    if DASHSCOPE_KEY and HAS_OPENAI:
        return "dashscope"
    if ollama_available():
        return "ollama"
    return "hf" # Always fallback to Public HF Inference API


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
               "max_tokens": 8192, "temperature": 0.7, "top_p": 0.9}
    headers = {"Authorization": f"Bearer {OLLAMA_KEY}", "Content-Type": "application/json"}
    with http.post(url, json=payload, headers=headers, stream=True, timeout=360) as resp:
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
                    yield delta
            except Exception:
                continue


# ═══════════════════════════════════════════════════════════════════════════
# STREAMING — Hugging Face Inference API (OpenAI-compat)
# ═══════════════════════════════════════════════════════════════════════════

def hf_stream(messages: list[dict], model: str):
    """Stream from HF Inference API using OpenAI-compat endpoint."""
    hf_model = model if "/" in model else HF_MODEL
    url = f"https://api-inference.huggingface.co/v1/chat/completions"
    payload = {
        "model": hf_model,
        "messages": messages,
        "stream": True,
        "max_tokens": 8192,
        "temperature": 0.7,
    }
    headers = {"Content-Type": "application/json"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    with http.post(url, json=payload, headers=headers, stream=True, timeout=360) as resp:
        if resp.status_code != 200:
            yield f"[HF ERROR {resp.status_code}]: {resp.text[:300]}"
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
                    yield delta
            except Exception:
                continue


# ═══════════════════════════════════════════════════════════════════════════
# STREAMING — DashScope (Alibaba)
# ═══════════════════════════════════════════════════════════════════════════

def dashscope_stream(messages: list[dict], model: str):
    """Stream from DashScope using the OpenAI SDK."""
    if not HAS_OPENAI or not DASHSCOPE_KEY:
        yield f"[ERROR] Missing DashScope configuration or openai python package."
        return

    client = OpenAI(
        api_key=DASHSCOPE_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            temperature=0.7,
            max_tokens=6000 # Dashscope preferred limit
        )
        for chunk in resp:
            content = chunk.choices[0].delta.content
            if content:
                yield content
    except Exception as e:
        # Check if limit was exceeded or max tokens error
        err_msg = str(e).lower()
        if "limit" in err_msg or "quota" in err_msg or "max_tokens" in err_msg:
            # Fallback to qwen-plus immediately
            fallback_model = "qwen-plus"
            yield f"\n\n\n[DASH_SCOPE_FALLBACK:{fallback_model}]\n\n\n"
            
            resp = client.chat.completions.create(
                model=fallback_model,
                messages=messages,
                stream=True,
                temperature=0.7,
                max_tokens=6000
            )
            for chunk in resp:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        else:
            yield f"[ERROR DashScope]: {err_msg}"


def auto_stream(messages: list[dict], model: str, backend: str):
    """Route to correct backend logic."""
    if model.startswith("qwen-") or "qwen3-max" in model or backend == "dashscope":
        yield from dashscope_stream(messages, model)
    elif backend == "hf":
        yield from hf_stream(messages, model)
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
    if backend == "dashscope":
        try:
            r = http.get("http://localhost:11434/api/tags", timeout=1)
            ollama_models = [m["name"] for m in r.json().get("models", [])]
        except:
            ollama_models = []
        return jsonify({"ok": True, "dashscope": True, "ollama": bool(ollama_models), "hf": True,
                        "model": "qwen-plus", "qwen_models": ollama_models})
    elif backend == "ollama":
        try:
            r = http.get("http://localhost:11434/api/tags", timeout=3)
            models = [m["name"] for m in r.json().get("models", [])]
            qwen = [m for m in models if "qwen" in m.lower()]
            return jsonify({"ok": True, "dashscope": False, "ollama": True, "hf": True,
                            "models": models, "qwen_models": qwen})
        except Exception as e:
            return jsonify({"ok": False, "dashscope": False, "ollama": False, "hf": True, "error": str(e)})
    else:
        return jsonify({"ok": True, "dashscope": False, "ollama": False, "hf": True,
                        "model": HF_MODEL, "qwen_models": []})


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
        return jsonify({"error": "No AI backend available. Start Ollama or set HF_TOKEN."}), 503

    def stream():
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

        messages = [{"role": "system", "content": full_system}]
        for m in engine.transcript_store.replay()[-6:]:
            if m.startswith("U:"):
                messages.append({"role": "user", "content": m[2:].strip()})
            elif m.startswith("A:"):
                messages.append({"role": "assistant", "content": m[2:].strip()})

        user_msg = build_topic_prompt(topic) if mode == "topic" else build_content_prompt(content, src_name)
        messages.append({"role": "user", "content": user_msg})
        history.add("messages", f"len={len(messages)} mode={mode} backend={backend}")

        # ── metadata event ───────────────────────────────────────────────
        yield f"data: {json.dumps({'type':'meta','session_id':engine.session_id,'model':model,'mode':mode,'backend':backend,'routes':[{'kind':m.kind,'name':m.name,'score':m.score} for m in matches[:4]]})}\\n\\n"

        # ── stream tokens ────────────────────────────────────────────────
        start  = time.time()
        chunks = []
        for chunk in auto_stream(messages, model, backend):
            if "[DASH_SCOPE_FALLBACK:" in chunk:
                new_model = chunk.split("[DASH_SCOPE_FALLBACK:")[1].split("]")[0]
                model = new_model
                yield f"data: {json.dumps({'type':'fallback','new_model':new_model})}\\n\\n"
                continue

            chunks.append(chunk)
            yield f"data: {json.dumps({'type':'chunk','text':chunk})}\\n\\n"

        elapsed  = round(time.time() - start, 1)
        raw_html = "".join(chunks).strip()

        # strip markdown fences if model wrapped output
        if raw_html.startswith("```html"):
            raw_html = raw_html[7:]
        if raw_html.startswith("```"):
            raw_html = raw_html[3:]
        if raw_html.endswith("```"):
            raw_html = raw_html[:-3]
        raw_html = raw_html.strip()

        # ── claw session update ──────────────────────────────────────────
        engine.submit_message(
            f"Notes: {topic or src_name}",
            matched_commands=tuple(m.name for m in matches if m.kind == "command"),
            matched_tools=tuple(m.name for m in matches if m.kind == "tool"),
        )
        engine.transcript_store.append(f"U: {user_msg[:300]}")
        engine.transcript_store.append(f"A: [HTML notes {len(raw_html)} chars]")
        engine.compact_messages_if_needed()
        session_path = engine.persist_session()
        history.add("done", f"elapsed={elapsed}s session={session_path}")

        # ── save HTML ────────────────────────────────────────────────────
        safe = "".join(c for c in (topic or src_name)[:40] if c.isalnum() or c in " -_").strip().replace(" ", "_")
        fname = f"{safe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        (NOTES_DIR / fname).write_text(raw_html, encoding="utf-8")

        yield f"data: {json.dumps({'type':'done','elapsed':elapsed,'note_file':fname,'session_id':engine.session_id})}\\n\\n"

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
            return jsonify({"ok": True, "message": "Ollama is already running!"})
    except:
        pass
        
    try:
        # Launch ollama detached
        subprocess.Popen(["ollama", "run", model])
        time.sleep(1) # wait briefly for server to bind
        return jsonify({"ok": True, "message": f"Successfully launched local instance of {model}!"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    backend = get_backend()
    print(f"\n{'═'*54}")
    print(f"  🎓  NotesMaster AI")
    print(f"  🤖  Model : {DEFAULT_MODEL}")
    print(f"  🔌  Backend: {backend.upper()} {'('+HF_MODEL+')' if backend=='hf' else ''}")
    print(f"  🌐  http://localhost:{PORT}")
    print(f"{'═'*54}\n")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
