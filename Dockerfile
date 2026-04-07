FROM python:3.11-slim

# ── system deps ──
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── work dir ──
WORKDIR /app

# ── copy the full repo (claw src + notesmaster) ──
COPY . .

# ── install Python deps ──
RUN pip install --no-cache-dir -r notesmaster/requirements.txt

# ── HF Spaces require port 7860 ──
ENV PORT=7860
EXPOSE 7860

# ── run ──
CMD ["python", "notesmaster/server.py"]
