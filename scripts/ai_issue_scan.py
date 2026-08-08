import os, json, glob, requests

GROQ_KEY = os.environ["GROQ_API_KEY"]
GH_TOKEN = os.environ["GITHUB_TOKEN"]
REPO = os.environ["GITHUB_REPOSITORY"]
MAX_ISSUES = 5

SCAN_PATHS = ["leanpass/*.py", "tests/*.py"]

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

Respond with ONLY a JSON array (no markdown fences), max {MAX_ISSUES} items:
[{{"title": "...", "body": "...", "file": "..."}}]

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
    requests.post(
        f"https://api.github.com/repos/{REPO}/issues",
        headers={
            "Authorization": f"Bearer {GH_TOKEN}",
            "Accept": "application/vnd.github+json",
        },
        json={
            "title": item["title"],
            "body": f"**File:** `{item.get('file', 'unknown')}`\n\n{item['body']}\n\n_Filed automatically by ai-issue-scan._",
            "labels": ["ai-scan"],
        },
        timeout=30,
    )

if __name__ == "__main__":
    code = gather_code()
    try:
        issues = ask_groq(code)
    except (json.JSONDecodeError, requests.RequestException) as e:
        print(f"Scan failed or returned unparseable output: {e}")
        raise SystemExit(0)  # don't fail the whole workflow over a bad LLM response

    for item in issues[:MAX_ISSUES]:
        create_issue(item)
        print(f"Created issue: {item['title']}")