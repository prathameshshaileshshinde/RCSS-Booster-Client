# Git & GitHub Workflow Guide

This document explains the Git workflow used in this project and teaches the core commands you need day-to-day.

---

## Core Concepts

| Term | Meaning |
|---|---|
| **Repository (repo)** | The project folder tracked by Git |
| **Commit** | A snapshot of your changes, saved permanently |
| **Branch** | An independent line of development |
| **Remote** | A copy of the repo on GitHub (or another server) |
| **Push** | Send your local commits to the remote |
| **Pull** | Download remote commits into your local repo |
| **Merge** | Combine two branches together |

---

## One-Time Setup

```bash
# Set your identity (used in commit messages)
git config --global user.name  "Your Name"
git config --global user.email "you@example.com"

# Optional: set VS Code as the default editor
git config --global core.editor "code --wait"
```

---

## Clone the Repository

```bash
# Download the repo to your machine
git clone https://github.com/<username>/nn_client.git

# Enter the project folder
cd nn_client
```

---

## Everyday Workflow

### 1. Check status before doing anything

```bash
git status          # see which files changed
git log --oneline   # see recent commits
```

### 2. Create a branch for new work

**Never commit directly to `master`.** Always create a branch first.

```bash
# Create and switch to a new branch
git checkout -b feature/vision-control

# Check which branch you're on
git branch
```

Branch naming conventions used in this project:
- `feature/<name>` — new feature
- `fix/<name>` — bug fix
- `docs/<name>` — documentation only
- `refactor/<name>` — code cleanup with no behaviour change

### 3. Make changes and commit

```bash
# Stage a specific file
git add nn_client.py

# Stage all changed files
git add .

# Commit with a clear message
git commit -m "feat: add world-model ball persistence"
```

**Good commit message format:**

```
<type>: <short description (50 chars max)>

Optional longer explanation of WHY the change was made,
not WHAT it does (the diff shows that).
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`

### 4. Push your branch to GitHub

```bash
# First push (sets upstream)
git push -u origin feature/vision-control

# Subsequent pushes on the same branch
git push
```

### 5. Open a Pull Request on GitHub

1. Go to the repo on GitHub
2. Click **"Compare & pull request"** next to your branch
3. Write a description of what you changed and why
4. Request a review (if working in a team)
5. Merge when approved

### 6. Pull latest changes from main

```bash
# Switch to master and update
git checkout master
git pull origin master
```

---

## Resolving Merge Conflicts

Conflicts happen when two people edit the same lines. Git marks them like this:

```python
<<<<<<< HEAD
goal_vel = np.array([1.0, 0.0, yaw_vel])   # your version
=======
goal_vel = np.array([0.8, 0.0, yaw_vel])   # teammate's version
>>>>>>> feature/slower-walk
```

To resolve:

1. Open the file and decide which version to keep (or combine them)
2. Delete all the `<<<<<<<`, `=======`, `>>>>>>>` markers
3. Save the file
4. Stage and commit the resolution:

```bash
git add nn_client.py
git commit -m "merge: resolve goal_vel conflict — keep 1.0 speed"
```

### Merge conflict tip

Pull from master frequently so conflicts stay small:

```bash
git checkout feature/my-branch
git merge master            # bring in latest master changes
# resolve any conflicts, then continue
```

---

## Useful Commands Reference

```bash
# See what changed in a file
git diff nn_client.py

# See changes in a specific commit
git show <commit-hash>

# Undo unstaged changes to a file
git checkout -- nn_client.py

# Undo the last commit (keep changes in working directory)
git reset --soft HEAD~1

# Stash changes temporarily (e.g. to switch branches)
git stash
git stash pop              # bring stashed changes back

# See all branches (local and remote)
git branch -a

# Delete a local branch (after merging)
git branch -d feature/old-feature

# Delete a remote branch
git push origin --delete feature/old-feature

# Tag a release
git tag -a v1.0 -m "Initial working version"
git push origin --tags
```

---

## Commit History: This Project

The commits in this repo follow the feature-by-feature structure:

```
feat: add Voronoi research document and README section
feat: add start_team.py cross-platform team launcher
feat: expand README with full system documentation
docs: add GIT_WORKFLOW.md guide
feat: add CSV position logging with game state and teammate data
feat: implement world-model ball persistence and head sweep search
feat: add SEEK→ORBIT→PUSH movement state machine
feat: add _parse_teammates() for multi-robot detection
feat: initial nn_client with ball tracking and locomotion policy
```

Each commit corresponds to one logical unit of work, making it easy to understand the project's history and revert individual features if needed.

---

## Pushing to GitHub (End-of-Day Checklist)

```bash
# 1. Make sure all changes are committed
git status

# 2. Push current branch
git push

# 3. If this was work on master directly (not recommended but happens)
git push origin master

# 4. Verify on GitHub that your commits appear
#    https://github.com/<username>/nn_client/commits/master
```

---

## Common Mistakes and Fixes

| Mistake | Fix |
|---|---|
| Committed to master by accident | `git reset --soft HEAD~1` then create a branch |
| Pushed sensitive data (e.g. API key) | Remove, force push, then rotate the key |
| Wrong commit message | `git commit --amend -m "correct message"` (before pushing) |
| Forgot to pull before starting | `git pull origin master` — may need to resolve conflicts |
| Need to undo a pushed commit | `git revert <hash>` — creates a new undo commit safely |
