#!/bin/bash

# 가상환경 자동 활성화
source venv/bin/activate

echo "====================================================="
echo "[DEBUG] Python Environment Check"
echo "1. 실행되는 파이썬 경로: $(which python)"  # <-- 진짜 venv 파이썬인지 확인!
echo "2. 파이썬 버전: $(python --version)"
echo "3. Torch 설치 경로: $(python -c 'import torch; print(torch.__file__)' 2>/dev/null || echo '❌ Torch 없음!')"
echo "====================================================="

# PYTHONPATH 깔끔하게 세팅
export PYTHONPATH="$(pwd)"
echo "[DEBUG] PYTHONPATH: $PYTHONPATH"

echo "====================================================="
echo "🐛 디버거 대기 중... VS Code에서 F5(Attach)를 누르세요!"
echo "====================================================="

FILE_DIR="train"
FILE_NAME="drifting_tarflow_two_dimensional_learnable_ratio_reg.py"

# 백슬래시 뒤에 공백 절대 없도록 수정 완료
./venv/bin/python -m debugpy --listen 0.0.0.0:5678 --wait-for-client \
    ${FILE_DIR}/${FILE_NAME} 