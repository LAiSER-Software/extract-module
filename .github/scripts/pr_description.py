import os

import google.generativeai as genai
import requests

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
REPO = os.environ["GITHUB_REPOSITORY"]
PR_NUMBER = os.environ["PR_NUMBER"]

# Read diff from file instead of env var to avoid "argument list too long" error
with open("pr.diff", "r", encoding="utf-8") as f:
    PR_DIFF = f.read()

# ── Generate description via Gemini ────────────────────────────────────────
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

prompt = f"""You are a senior software engineer. Given the following git diff, write a concise PR description.

Format:
## What changed
- bullet points of what changed

## Why
- reason for the change

## Notes
- anything reviewers should know (optional)

Keep it short and technical. No fluff.

Git diff:
{PR_DIFF[:12000]}
"""

response = model.generate_content(prompt)
description = response.text.strip()

# ── Update PR description via GitHub API ───────────────────────────────────
url = f"https://api.github.com/repos/{REPO}/pulls/{PR_NUMBER}"
headers = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
}
res = requests.patch(url, headers=headers, json={"body": description})

if res.status_code == 200:
    print("PR description updated successfully.")
else:
    print(f"Failed to update PR description: {res.status_code} {res.text}")
    exit(1)
