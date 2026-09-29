FROM node:20-bookworm-slim AS frontend-builder

WORKDIR /app/webapp

COPY webapp/package.json webapp/package-lock.json ./
RUN npm ci

COPY webapp/ ./
RUN npm run build


FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DEVICE=cpu \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    TENNIS_LOG_RETENTION_DAYS=7

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN grep -Ev '^(torch|torchvision)([<>=!~].*)?$' requirements.txt > /tmp/requirements-no-torch.txt \
    && pip install --upgrade pip wheel "setuptools<81" \
    && pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.1.2 torchvision==0.16.2 \
    && pip install --find-links https://download.openmmlab.com/mmcv/dist/cpu/torch2.1.0/index.html \
        mmcv==2.1.0 \
    && pip install --no-build-isolation chumpy==0.70 \
    && pip install -r /tmp/requirements-no-torch.txt

COPY app.py ./
COPY extract_pose_features.py summarize_pose_features.py visualize_pose.py visualize_pose_video.py ./
COPY tennis_coach ./tennis_coach
COPY models/detector ./models/detector
COPY rtmpose-l_8xb256-420e_coco-256x192.py ./
COPY rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192-1352a4d2_20230127.pth ./
COPY .env* ./
RUN test -f .env || cp .env.example .env
COPY --from=frontend-builder /app/webapp/dist ./webapp/dist

RUN mkdir -p /app/analysis_tasks /app/logs

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
