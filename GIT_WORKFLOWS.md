# Git Workflows for ai-vocals-studio

## Repository Status
This repository is now private and configured for collaborative development.

## Commit Workflow
1. **Stage Changes**
   ```bash
   git add .
   # or stage specific files
   git add path/to/file.py
   ```

2. **Commit Changes**
   ```bash
   git commit -m "Descriptive commit message"
   ```

3. **Check Status**
   ```bash
   git status
   ```

## Push Workflow
1. **Push to Remote**
   ```bash
   git push origin main
   ```

2. **Force Push (if needed)**
   ```bash
   git push origin main --force-with-lease
   ```

## Pull Workflow
1. **Pull Latest Changes**
   ```bash
   git pull origin main
   ```

2. **Pull with Rebase**
   ```bash
   git pull --rebase origin main
   ```

## Sync Workflow
1. **Fetch All Changes**
   ```bash
   git fetch --all
   ```

2. **Sync with Remote**
   ```bash
   git pull origin main
   git push origin main
   ```

## Deploy Workflow
1. **Check Current Status**
   ```bash
   git status
   ```

2. **Pull Latest Changes**
   ```bash
   git pull origin main
   ```

3. **Deploy (if applicable)**
   ```bash
   # Add your deployment commands here
   ```

## Additional Commands
- **View Commit History**
  ```bash
  git log --oneline
  ```

- **View Branch Status**
  ```bash
  git branch -a
  ```

- **View Remote Status**
  ```bash
  git remote -v