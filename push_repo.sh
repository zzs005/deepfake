#!/bin/bash
# Push to GitHub - force approach
set -e

cd "/c/Users/szh/Desktop/机器学习大作业"
source ~/.hermes/.env

REPO_URL="https://zzs005:${GITHUB_TOKEN}@github.com/zzs005/deepfake.git"

# Clean slate .git
rm -rf ".git"

git init
git lfs install
git lfs track "models/*.pth"
git lfs track "models/*.joblib"

cat > .gitattributes << 'GITATTR'
*.pth filter=lfs diff=lfs merge=lfs -text
*.joblib filter=lfs diff=lfs merge=lfs -text
models/* filter=lfs diff=lfs merge=lfs -text
GITATTR

git remote add origin "$REPO_URL"
git fetch origin

# Force checkout to remote main (overwrite any conflicting local files)
git checkout -f -B main origin/main

# Now add all our local files on top
git add -A

if git diff --cached --quiet; then
    echo "No new files to add"
else
    git commit -m "添加模型参数文件、requirements.txt和readme说明"
fi

# Force push is needed since we're reorganizing
git push -u origin main

echo "=== 推送完成 ==="
