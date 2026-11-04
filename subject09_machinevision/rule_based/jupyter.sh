#!/usr/bin/env bash
# 교과 9 JupyterLab 실행 — 프로젝트 최상위 폴더 기준으로 켜고 Windows 기본 브라우저로 엽니다.
# 실행: ~/ai_vision/subject9_machinevision/jupyter.sh   끄기: 이 터미널에서 Ctrl+C
# 브라우저가 안 열리면 터미널에 찍힌 http://localhost:8888/lab?token=... 주소를 Ctrl+클릭하거나 복사해서 엽니다.
cd "$(dirname "$0")" || exit 1
source .venv/bin/activate
# WSL에는 브라우저가 없어서 Windows 탐색기에 주소를 넘겨 기본 브라우저로 엽니다.
# use_redirect_file=False: 임시 html 파일 대신 http 주소(토큰 포함)를 바로 넘기기 위해 필요
export BROWSER=/mnt/c/Windows/explorer.exe
exec jupyter lab --ServerApp.use_redirect_file=False --port 8888 "$@"
