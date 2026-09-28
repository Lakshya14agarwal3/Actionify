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

ACTION_EXAMPLES = [
    "Send the project update to the client",
    "Someone will send the budget tomorrow",
    "Someone needs to complete this task",
    "Schedule the next team meeting",
    "Prepare the presentation before Friday",
    "Review the report and share feedback",
    "Follow up with the vendor",
    "Update the dashboard with the latest figures",
    "Assign someone to finish this task",
    "Lakshya will finalize the UI by October 5",
    "Priya will draft the roadmap by October 3",
    "Rahul will analyze GPU allocation by October 7",
    "Sneha will prepare campaign assets by October 10",
]

DISCUSSION_EXAMPLES = [
    "The team discussed the monthly results",
    "Sales increased this quarter",
    "The project status was shared",
    "Everyone agreed with the proposed timeline",
    "The client explained their feedback",
    "The team reviewed the prototype progress",
    "Backend integration is complete",
    "Technical challenges were highlighted",
    "Sneha outlined plans for LinkedIn campaigns and beta launch",
    "The group agreed to prioritize workflow automation for Q4",
]

LOCAL_MODEL_PATH = BASE_DIR / "backend" / "models" / "all-MiniLM-L6-v2"

embedding_model = SentenceTransformer(str(LOCAL_MODEL_PATH))
reference_texts = ACTION_EXAMPLES + DISCUSSION_EXAMPLES
reference_embeddings = embedding_model.encode(reference_texts, normalize_embeddings=True)

# Tiny trained head (backend/train_classifier.py): logistic regression on frozen
# MiniLM embeddings. ~2KB, committed to git, loaded offline. Falls back to
# prototype matching if the file is missing.
ACTION_THRESHOLD = 0.5
_action_head = None
try:
    _head_file = BASE_DIR / "backend" / "action_head.npz"
    if _head_file.exists():
        _head_data = np.load(_head_file)
        _action_head = (_head_data["coef"].ravel(), float(_head_data["intercept"].ravel()[0]))
except Exception:
    _action_head = None


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
    if _action_head is not None:
        return _head_probability(task_embedding) >= ACTION_THRESHOLD
    scores = reference_embeddings @ task_embedding
    action_score = float(max(scores[:len(ACTION_EXAMPLES)]))
    discussion_score = float(max(scores[len(ACTION_EXAMPLES):]))
    return action_score > discussion_score + 0.02 and action_score >= 0.25


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
