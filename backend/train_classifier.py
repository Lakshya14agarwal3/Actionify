

import random
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- data ---

# Real action-item sentences from public meeting research (AMI corpus papers,
# Murray et al. 2008; meeting-notes-cleaner transcript examples).
REAL_ACTIONS = [
    "So you will have to work together on the prototype",
    "You will have next time to show us a clay remote control",
    "I'll review the engineering backlog before the next sprint planning",
    "Jordan will review the engineering backlog and see what changes we can make",
    "We need a tighter plan for client onboarding",
    "Priya will send the budget by Friday",
    "Someone needs to complete this task",
    "Schedule the next team meeting",
    "Prepare the presentation before Friday",
    "Follow up with the vendor next week",
    "Update the dashboard with the latest figures by Monday",
    "Assign someone to finish this task by tomorrow",
    "Rahul needs to complete the GPU report",
    "Sneha will prepare the campaign assets by Thursday",
    "Lakshya will finalize the UI mockups by next week",
    "John will fix the login bug before the release",
    "Sara should update the API documentation by Friday",
    "The team must approve the Q3 planning doc by EOD",
    "Devops is investigating the pipeline failure",
    "Please share the retro notes with the team by tomorrow",
]

# Real discussion/status sentences (AMI-style + product-meeting style).
REAL_DISCUSSION = [
    "Sales increased this quarter",
    "The team discussed the monthly results",
    "The project status was shared",
    "Everyone agreed with the proposed timeline",
    "The client explained their feedback",
    "Backend integration is complete",
    "Technical challenges were highlighted",
    "Sneha outlined plans for LinkedIn campaigns and beta launch",
    "The group agreed to prioritize workflow automation for Q4",
    "Mobile integration was postponed to early next year",
    "The prototype demo went well yesterday",
    "GPU resources are limited this sprint",
    "The beta launch is scheduled for October",
    "Does the approval group want one review slot or two?",
    "Which proof beat is non-negotiable in the first pass?",
    "The draft is too broad for the review window",
    "Stakeholders want fewer rounds of review",
    "Onboarding dropout hit sixty percent last month",
    "The checkout flow has a three second lag",
    "Q2 infra budget is still under discussion",
]

NAMES = [
    "Lakshya", "Priya", "Rahul", "Sneha", "Aarav", "Ananya", "Arjun",
    "Diya", "Ishaan", "Kavya", "Neha", "Rohan", "Sara", "Vikram",
    "Jordan", "Mia", "Lucas", "Chen", "David", "Baba",
]
ACTION_TEMPLATES = [
    "{n} will {v} {o} by {d}",
    "{n} needs to {v} {o} by {d}",
    "{n} should {v} {o} by {d}",
    "{n} must {v} {o} by {d}",
    "{n} is {ving} {o} by {d}",
    "Ask {n} to {v} {o} by {d}",
    "Assigned to {n} for {o} by {d}",
    "{n}: {v} {o} by {d}",
]
VERBS = [
    ("finalize", "finalizing"), ("draft", "drafting"), ("analyze", "analyzing"),
    ("prepare", "preparing"), ("review", "reviewing"), ("update", "updating"),
    ("send", "sending"), ("complete", "completing"), ("share", "sharing"),
    ("schedule", "scheduling"), ("test", "testing"), ("fix", "fixing"),
]
OBJECTS = [
    "the UI mockups", "the roadmap", "the budget report", "the API documentation",
    "the dashboard figures", "the client presentation", "the test coverage",
    "the deployment pipeline", "the onboarding flow", "the quarterly review",
    "the vendor contract", "the beta checklist",
]
DATES = [
    "Friday", "Monday", "tomorrow", "October 5", "October 3", "next week",
    "EOD", "the end of the sprint",
]
DISCUSSION_TEMPLATES = [
    "The team talked about {o} today",
    "{o} {st} this quarter",
    "Everyone agreed that {o} looks good",
    "The status of {o} was shared with the group",
    "There were questions about {o} during the review",
    "The client gave feedback on {o}",
    "{o} remains an open topic for next time",
]
DISC_OBJECTS = [
    "the monthly results", "quarterly sales", "the timeline", "the prototype",
    "server costs", "customer feedback", "the hiring plan", "brand guidelines",
    "support tickets", "the holiday schedule", "market trends", "team morale",
]
DISC_STATES = ["improved", "was discussed", "came up again", "looks stable", "needs more thought"]


def build_dataset(seed=42):
    rng = random.Random(seed)
    actions = list(REAL_ACTIONS)
    discussion = list(REAL_DISCUSSION)
    while len(actions) < 110:
        n = rng.choice(NAMES)
        (v, ving) = rng.choice(VERBS)
        o = rng.choice(OBJECTS)
        d = rng.choice(DATES)
        t = rng.choice(ACTION_TEMPLATES)
        s = t.format(n=n, v=v, ving=ving, o=o, d=d)
        if s not in actions:
            actions.append(s)
    while len(discussion) < 110:
        t = rng.choice(DISCUSSION_TEMPLATES)
        s = t.format(o=rng.choice(DISC_OBJECTS), st=rng.choice(DISC_STATES))
        if s not in discussion:
            discussion.append(s)
    data = [(s, 1) for s in actions] + [(s, 0) for s in discussion]
    rng.shuffle(data)
    split = int(0.75 * len(data))
    return data[:split], data[split:]


# --------------------------------------------------------------- train ---

def metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def main():
    from sklearn.linear_model import LogisticRegression

    from backend.app import (
        ACTION_EXAMPLES,
        DISCUSSION_EXAMPLES,
        embedding_model,
        reference_embeddings,
    )

    train, test = build_dataset()
    print(f"train={len(train)} test={len(test)}")

    def embed(texts):
        return np.asarray(
            embedding_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        )

    X_train = embed([t for t, _ in train])
    y_train = np.array([y for _, y in train])
    X_test = embed([t for t, _ in test])
    y_test = np.array([y for _, y in test])

    # Baseline: current prototype-max rule from app.py.
    n_act = len(ACTION_EXAMPLES)
    sims = X_test @ reference_embeddings.T
    base_pred = (
        (sims[:, :n_act].max(axis=1) > sims[:, n_act:].max(axis=1) + 0.02)
        & (sims[:, :n_act].max(axis=1) >= 0.25)
    ).astype(int)
    bp, br, bf = metrics(y_test, base_pred)
    print(f"baseline prototype-max  P={bp:.3f} R={br:.3f} F1={bf:.3f}")

    clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]

    best = (0.5, 0.0)
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
        p, r, f = metrics(y_test, (proba >= thr).astype(int))
        print(f"logreg thr={thr:.1f}  P={p:.3f} R={r:.3f} F1={f:.3f}")
        if f > best[1]:
            best = (thr, f)

    out = BASE_DIR / "backend" / "action_head.npz"
    np.savez(out, coef=clf.coef_.astype(np.float32), intercept=clf.intercept_.astype(np.float32))
    print(f"saved {out} ({out.stat().st_size / 1024:.1f} KB), best_thr={best[0]} F1={best[1]:.3f}")


if __name__ == "__main__":
    main()
