#!/usr/bin/env bash
# 一键推送到 GitHub（需先在 GitHub 网页上创建一个空仓库，不要勾选 README）
#
# 用法：
#   ./deploy.sh https://github.com/<用户名>/<仓库名>.git
#
# 推送成功后：
#   仓库 Settings → Pages → Build and deployment → Source 选 "GitHub Actions"
#   然后访问 https://<用户名>.github.io/<仓库名>/
set -euo pipefail

REMOTE="${1:-}"
if [ -z "$REMOTE" ]; then
  echo "用法: ./deploy.sh https://github.com/<用户名>/<仓库名>.git"
  exit 1
fi

git remote remove origin 2>/dev/null || true
git remote add origin "$REMOTE"
git branch -M main

echo "→ 推送到 $REMOTE ..."
git push -u origin main

echo ""
echo "✅ 推送完成"
echo "下一步：仓库 Settings → Pages → Source 选『GitHub Actions』"
echo "然后等 Actions 跑完，打开 https://<用户名>.github.io/<仓库名>/"
