import re
import sqlite3
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = BASE_DIR / "database" / "meetings.db"

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "frontend" / "templates"),
    static_folder=str(BASE_DIR / "frontend" / "static"),
)

LOCAL_MODEL_PATH = BASE_DIR / "backend" / "models" / "all-MiniLM-L6-v2"

embedding_model = SentenceTransformer(str(LOCAL_MODEL_PATH))

# Trained classifier head (backend/train_classifier.py): logistic regression on
# frozen MiniLM embeddings. ~2KB, committed to git, loaded offline.
ACTION_THRESHOLD = 0.5
_head_data = np.load(BASE_DIR / "backend" / "action_head.npz")
_action_head = (_head_data["coef"].ravel(), float(_head_data["intercept"].ravel()[0]))


def _head_probability(task_embedding):
    coef, intercept = _action_head
    logit = float(np.dot(coef, task_embedding) + intercept)
    return 1.0 / (1.0 + np.exp(-logit))

DATE_BY_RE = re.compile(
    r"\bby\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?"
    r"|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
    r"|monday|tuesday|wednesday|thursday|friday|tomorrow|today|\d)",
    re.IGNORECASE,
)
ACTION_VERB_RE = re.compile(
    r"\b(will|need(?:s)?\s+to|follow\s*up|finaliz\w*|draft\w*|prepar\w*|analys\w*|analyz\w*"
    r"|complet\w*|send\w*|schedul\w*|assign\w*|updat\w*|review\w*|creat\w*|shar\w*)\b",
    re.IGNORECASE,
)
OWNER_VERB_RE = re.compile(
    r"\b([A-Z][a-z]+)\s+(?:will\s+)?(finalizing|finalize|drafting|draft|analyzing|analysing|"
    r"analyze|analyse|preparing|prepare|send|complete|review|schedule|update|assign|share|create)\b"
)
OWNER_PREFIX_RE = re.compile(r"^([A-Z][a-z]+)\s*[:,–—-]\s+\S")
OWNER_ASSIGNED_RE = re.compile(
    r"\bassign(?:ed)?\s+(?:to\s+)?([A-Z][a-z]+)\b|\b([A-Z][a-z]+)\s+is\s+assign(?:ed)?\b",
    re.IGNORECASE,
)


def setup_database():
    DATABASE_PATH.parent.mkdir(exist_ok=True)
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS meeting_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                notes TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def save_notes(notes):
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("INSERT INTO meeting_notes (notes) VALUES (?)", (notes,))


def is_action_item(task):
    task = task.strip()
    if len(task.split()) < 3:
        return False
    # Explicit action pattern: "Name ... action-verb ... by <date>"
    # e.g. "Lakshya finalizing the UI by October 5"
    if DATE_BY_RE.search(task) and ACTION_VERB_RE.search(task):
        return True
    task_embedding = embedding_model.encode(task, normalize_embeddings=True)
    return _head_probability(task_embedding) >= ACTION_THRESHOLD


def find_owner(task):
    match = OWNER_VERB_RE.search(task)
    if match:
        return match.group(1)
    match = OWNER_PREFIX_RE.search(task)
    if match:
        return match.group(1)
    match = OWNER_ASSIGNED_RE.search(task)
    if match:
        name = match.group(1) or match.group(2)
        if name.lower() not in {
            "this", "that", "it", "the", "task", "tasks", "work", "team", "to",
        }:
            return name[0].upper() + name[1:]
    first_word = task.split()[0].rstrip(":,")
    no_owner_words = {
        "please", "we", "the", "schedule", "review", "follow", "prepare", "send", "update",
        "in", "on", "key", "technical", "backend", "frontend", "group", "team", "project",
    }

    if first_word.lower() in no_owner_words:
        return "Unassigned"
    # Only treat capitalized words as names, otherwise Unassigned
    if not first_word[0].isupper():
        return "Unassigned"
    return first_word


def split_candidates(sentence):
    """Split 'Key action items include A, B, and C' into [A, B, C]."""
    match = re.search(r"\binclude[s]?\b[:\s]+(.+)", sentence, flags=re.IGNORECASE)
    if match and ("," in match.group(1) or " and " in match.group(1).lower()):
        parts = re.split(r";|,|\s+and\s+", match.group(1))
        return [part.strip() for part in parts if part.strip()]
    return [sentence]


def make_action_plan(notes):
    actions = []

    for line in notes.splitlines():
        line = line.strip()
        if not line:
            continue
        sentences = re.split(r"(?<=[.!?])\s+", line)
        for sentence in sentences:
            sentence = sentence.strip(" -•\t")
            if not sentence:
                continue
            for task in split_candidates(sentence):
                task = re.sub(r"^(and|or)\s+", "", task.strip(" -•\t.,"), flags=re.IGNORECASE)
                if not task or not is_action_item(task):
                    continue

                owner = find_owner(task)
                if owner == "Unassigned":
                    continue
                actions.append({"owner": owner, "task": task})

    return actions


@app.route("/", methods=["GET", "POST"])
def home():
    notes = ""
    actions = None

    if request.method == "POST":
        notes = request.form.get("notes", "").strip()
        if notes:
            save_notes(notes)
            actions = make_action_plan(notes)
        else:
            actions = []

    return render_template("index.html", notes=notes, actions=actions)


setup_database()


if __name__ == "__main__":
    app.run(debug=True)
