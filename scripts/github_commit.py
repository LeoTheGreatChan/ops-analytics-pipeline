"""
GitHub commit-on-material-change.

Render's free tier filesystem is ephemeral: a cold start boots a fresh
container from the last GitHub deploy, discarding anything written to disk
at runtime. This means the change-detection snapshot and the refreshed
data JSON files were silently lost on every cold start, undermining both
the "remember the last run" logic and the "visitors see current data"
promise.

Fix: when /api/refresh finds a material change, push the six updated files
straight back to GitHub via the Contents API. Since Render auto-deploys on
every push to main, the next cold start boots from a container that
already has this data baked in, no separate persistence layer needed, git
itself becomes the durable store.

Deliberately NOT called on non-material refreshes: committing on every
single refresh (most of which find nothing new) would spam the repo
history for no reason. Only genuine findings are worth a commit.
"""
import os
import base64
import requests

GITHUB_OWNER = os.environ.get('GITHUB_REPO_OWNER', 'LeoTheGreatChan')
GITHUB_REPO = os.environ.get('GITHUB_REPO_NAME', 'ops-analytics-pipeline')
GITHUB_BRANCH = os.environ.get('GITHUB_BRANCH', 'main')

FILES_TO_COMMIT = {
    'data/processed/kpis.json': 'kpis.json',
    'data/processed/segments.json': 'segments.json',
    'data/processed/insights_customer_satisfaction.json': 'insights_customer_satisfaction.json',
    'data/processed/insights_cost_reduction.json': 'insights_cost_reduction.json',
    'data/processed/forecast.json': 'forecast.json',
    'data/processed/findings_prior_snapshot.json': 'findings_prior_snapshot.json',
}


def _headers():
    token = os.environ.get('GITHUB_TOKEN')
    if not token:
        raise RuntimeError('GITHUB_TOKEN not set — cannot commit to GitHub')
    return {'Authorization': f'token {token}', 'Accept': 'application/vnd.github+json'}


def _get_file_sha(repo_path):
    """GitHub's Contents API requires the current file's SHA to update it
    (this is how it prevents accidental overwrites of concurrent edits).
    Returns None if the file doesn't exist yet in the repo."""
    url = f'https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/contents/{repo_path}'
    resp = requests.get(url, headers=_headers(), params={'ref': GITHUB_BRANCH}, timeout=15)
    if resp.status_code == 200:
        return resp.json()['sha']
    return None


def _commit_file(repo_path, local_path, message):
    with open(local_path, 'rb') as f:
        content_b64 = base64.b64encode(f.read()).decode('utf-8')

    sha = _get_file_sha(repo_path)
    url = f'https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/contents/{repo_path}'
    payload = {'message': message, 'content': content_b64, 'branch': GITHUB_BRANCH}
    if sha:
        payload['sha'] = sha  # required when updating an existing file

    resp = requests.put(url, headers=_headers(), json=payload, timeout=15)
    resp.raise_for_status()
    return resp.json()


def commit_updated_data_to_github(data_dir, reason):
    """
    data_dir: local directory where the freshly-regenerated JSON files live
              (same DATA_DIR api/app.py already uses)
    reason: the change_detection reason string, used in the commit message
            so the git history itself documents why each commit happened

    Returns a dict of {repo_path: result} for files actually committed.
    Silently skips any file that doesn't exist locally (e.g. if a step
    upstream failed to produce it) rather than raising, since a partial
    commit is better than the whole refresh failing over one missing file.
    """
    message = f'Automated refresh (n8n): {reason}'
    results = {}
    for repo_path, filename in FILES_TO_COMMIT.items():
        local_path = os.path.join(data_dir, filename)
        if os.path.exists(local_path):
            results[repo_path] = _commit_file(repo_path, local_path, message)
    return results
