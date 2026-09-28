# Actionify

A small Flask project that turns meeting-note lines into a simple action plan.

## Project folders

- `frontend/static` contains the CSS.
- `frontend/templates` contains the HTML page.
- `backend` contains the Flask app.
- `database` is where SQLite saves submitted notes.

## Run it

Create and activate a virtual environment in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python backend\app.py
```

Open `http://127.0.0.1:5000` in a browser.

Enter one action per line. Lines with terms such as `will`, `need to`, or `follow up` are added to the action plan.
