import sqlite3
from pathlib import Path

from flask import Flask, render_template, request


BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = BASE_DIR / "database" / "meetings.db"

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "frontend" / "templates"),
    static_folder=str(BASE_DIR / "frontend" / "static"),
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


def make_action_plan(notes):
    action_words = ("will", "need to", "follow up", "send", "prepare", "review", "schedule")
    actions = []

    for line in notes.splitlines():
        task = line.strip(" -•\t")
        if not task or not any(word in task.lower() for word in action_words):
            continue

        first_word = task.split()[0] if len(task.split()) > 1 else ""
        owner = first_word.rstrip(":,") or "Unassigned"
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
