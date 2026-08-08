import os, json, glob, re, difflib, requests

GROQ_KEY = os.environ["GROQ_API_KEY"]
GH_TOKEN = os.environ["GITHUB_TOKEN"]
REPO = os.environ["GITHUB_REPOSITORY"]  # e.g. "Terminay/leanpass"
MAX_ISSUES = 5
DUPLICATE_THRESHOLD = 0.75  # similarity ratio above which we treat titles as "the same issue"

SCAN_PATHS = ["leanpass/*.py", "tests/*.py"]

ALLOWED_LABELS = {
    "bug": "d73a4a",
    "enhancement": "a2eeef",
    "docs": "0075ca",
    "performance": "fbca04",
    "tests": "c5def5",
}

GH_HEADERS = {
    "Authorization": f"Bearer {GH_TOKEN}",
    "Accept": "application/vnd.github+json",
}


def normalize(title):
    return re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()


def ensure_labels():
    """Create any labels that don't exist yet so issue creation doesn't 422."""
    existing = {
        l["name"]
        for l in requests.get(
            f"https://api.github.com/repos/{REPO}/labels",
            headers=GH_HEADERS,
            params={"per_page": 100},
            timeout=30,
        ).json()
    }
    for name, color in ALLOWED_LABELS.items():
        if name not in existing:
            requests.post(
                f"https://api.github.com/repos/{REPO}/labels",
                headers=GH_HEADERS,
                json={"name": name, "color": color},
                timeout=30,
            )


def get_existing_titles():
    """Pull normalized titles of every issue currently in the repo (open or closed)."""
    titles = []
    page = 1
    while True:
        resp = requests.get(
            f"https://api.github.com/repos/{REPO}/issues",
            headers=GH_HEADERS,
            params={"state": "all", "per_page": 100, "page": page},
            timeout=30,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        titles.extend(normalize(i["title"]) for i in batch if "pull_request" not in i)
        if len(batch) < 100:
            break
        page += 1
    return titles


def is_duplicate(title, existing_titles):
    norm = normalize(title)
    for existing in existing_titles:
        if norm == existing:
            return True
        if difflib.SequenceMatcher(None, norm, existing).ratio() >= DUPLICATE_THRESHOLD:
            return True
    return False


def gather_code():
    chunks = []
    for pattern in SCAN_PATHS:
        for path in glob.glob(pattern):
            with open(path, "r", encoding="utf-8") as f:
                chunks.append(f"### {path}\n{f.read()}")
    return "\n\n".join(chunks)


def ask_groq(code):
    prompt = f"""You are reviewing a small NumPy-only autodiff library.
Scan the code below for real issues: missing edge-case handling,
undocumented public methods, numerically unstable ops (unguarded
log/exp/div), missing broadcasting checks, and gaps between typical
library expectations and actual behavior. Skip style nits.

For each issue, classify it with exactly one label from this set:
bug, enhancement, docs, performance, tests

Respond with ONLY a JSON array (no markdown fences), max {MAX_ISSUES} items:
[{{"title": "...", "body": "...", "file": "...", "label": "bug"}}]

CODE:
{code[:20000]}
"""
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_KEY}"},
        json={
            "model": "openai/gpt-oss-120b",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        },
        timeout=60,
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"].strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(text)


def create_issue(item):
    label = str(item.get("label", "")).strip().lower()
    if label not in ALLOWED_LABELS:
        label = "enhancement"  # safe default if Groq returns something unexpected

    requests.post(
        f"https://api.github.com/repos/{REPO}/issues",
        headers=GH_HEADERS,
        json={
            "title": item["title"],
            "body": f"**File:** `{item.get('file', 'unknown')}`\n\n{item['body']}\n\n_Filed automatically by ai-issue-scan._",
            "labels": [label],
        },
        timeout=30,
    )


if __name__ == "__main__":
    ensure_labels()
    existing_titles = get_existing_titles()

    code = gather_code()
    try:
        issues = ask_groq(code)
    except (json.JSONDecodeError, requests.RequestException) as e:
        print(f"Scan failed or returned unparseable output: {e}")
        raise SystemExit(0)  # don't fail the whole workflow over a bad LLM response

    created = 0
    for item in issues:
        if created >= MAX_ISSUES:
            break
        if is_duplicate(item["title"], existing_titles):
            print(f"Skipped (duplicate): {item['title']}")
            continue
        create_issue(item)
        existing_titles.append(normalize(item["title"]))  # avoid dupes within the same run too
        created += 1
        print(f"Created issue: {item['title']} [{item.get('label', 'enhancement')}]")

    print(f"Done. {created} new issue(s) created, {len(issues) - created} skipped.")
