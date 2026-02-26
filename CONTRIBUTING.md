# Contributing to TemGymCore

Thank you for your interest in contributing! This guide explains how to fork the repository, create a branch, and submit your changes.

## Forking and Branching

### 1. Fork the repository

1. Navigate to [https://github.com/TemGym/TemGymCore](https://github.com/TemGym/TemGymCore) in your browser.
2. Click the **Fork** button in the top-right corner.
3. Select your GitHub account as the destination. GitHub will create a copy of the repository at `https://github.com/<your-username>/TemGymCore`.

### 2. Clone your fork locally

```bash
git clone https://github.com/<your-username>/TemGymCore.git
cd TemGymCore
```

### 3. Add the upstream remote

Keep your fork up to date with the original repository by adding it as a remote:

```bash
git remote add upstream https://github.com/TemGym/TemGymCore.git
```

Verify the remotes:

```bash
git remote -v
# origin    https://github.com/<your-username>/TemGymCore.git (fetch)
# origin    https://github.com/<your-username>/TemGymCore.git (push)
# upstream  https://github.com/TemGym/TemGymCore.git (fetch)
# upstream  https://github.com/TemGym/TemGymCore.git (push)
```

### 4. Create a new branch

Always create a new branch for your changes rather than working on `main`:

```bash
git checkout -b my-feature-branch
```

Use a descriptive branch name, e.g. `fix-lens-propagation` or `add-new-component`.

### 5. Make your changes and commit

```bash
# ... edit files ...
git add .
git commit -m "Short description of your change"
```

### 6. Push your branch to your fork

```bash
git push origin my-feature-branch
```

### 7. Open a Pull Request

1. Go to your fork on GitHub (`https://github.com/<your-username>/TemGymCore`).
2. Click **Compare & pull request** next to your branch.
3. Fill in the title and description, then click **Create pull request**.

## Keeping your fork up to date

Before starting new work, sync your fork's `main` branch with upstream:

```bash
git checkout main
git fetch upstream
git merge upstream/main
git push origin main
```

## Development Setup

Install the package in editable mode with development dependencies:

```bash
pip install -e ".[dev]"
```

Run the test suite:

```bash
pytest
```

Run the linter:

```bash
flake8 src/
```

## Questions

If you have questions, feel free to open an [issue](https://github.com/TemGym/TemGymCore/issues).
