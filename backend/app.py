import sqlite3
from pathlib import Path

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
]

DISCUSSION_EXAMPLES = [
    "The team discussed the monthly results",
    "Sales increased this quarter",
    "The project status was shared",
    "Everyone agreed with the proposed timeline",
    "The client explained their feedback",
]

LOCAL_MODEL_PATH = BASE_DIR / "backend" / "models" / "all-MiniLM-L6-v2"

embedding_model = SentenceTransformer(str(LOCAL_MODEL_PATH))
reference_texts = ACTION_EXAMPLES + DISCUSSION_EXAMPLES
reference_embeddings = embedding_model.encode(reference_texts, normalize_embeddings=True)


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
    task_embedding = embedding_model.encode(task, normalize_embeddings=True)
    scores = reference_embeddings @ task_embedding
    action_score = max(scores[:len(ACTION_EXAMPLES)])
    discussion_score = max(scores[len(ACTION_EXAMPLES):])
    return action_score > discussion_score and action_score >= 0.25


def find_owner(task):
    first_word = task.split()[0].rstrip(":,")
    no_owner_words = {"please", "we", "the", "schedule", "review", "follow", "prepare", "send", "update"}

    if first_word.lower() in no_owner_words:
        return "Unassigned"
    return first_word


def make_action_plan(notes):
    actions = []

    for line in notes.splitlines():
        task = line.strip(" -•\t")
        if not task or not is_action_item(task):
            continue

        actions.append({"owner": find_owner(task), "task": task})

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
