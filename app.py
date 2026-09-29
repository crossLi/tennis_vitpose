import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import requests
import torch
from fastapi import Body, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision.ops import nms

from tennis_coach.kimi_client import (
    analyze_tennis_video,
    classify_tennis_video,
    derive_rule_based_score,
    normalize_analysis_payload,
    parse_json_from_text,
)
from tennis_coach.logging_utils import (
    ensure_logging_configured,
    get_app_logger,
    get_task_logger,
    log_with_task,
)

ensure_logging_configured(os.getenv("TENNIS_LOG_LEVEL", "INFO"))
app_logger = get_app_logger("app")

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - optional dependency
    YOLO = None

try:
    from mmpose.apis import inference_topdown, init_model
except ImportError:  # pragma: no cover - optional dependency
    inference_topdown = None
    init_model = None


BASE_DIR = Path(__file__).resolve().parent
ANALYSIS_TASKS_DIR = BASE_DIR / "analysis_tasks"
FRONTEND_DIST_DIR = BASE_DIR / "webapp" / "dist"
DEFAULT_LOCAL_RTMPOSE_CONFIG = BASE_DIR / "rtmpose-l_8xb256-420e_coco-256x192.py"
DEFAULT_LOCAL_RTMPOSE_CHECKPOINT = (
    BASE_DIR / "rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192-1352a4d2_20230127.pth"
)
DEFAULT_LOCAL_DETECTOR_DIR = BASE_DIR / "models" / "detector"
DEFAULT_LOCAL_YOLO_CANDIDATES = [
    DEFAULT_LOCAL_DETECTOR_DIR / "yolo26m.pt",
    DEFAULT_LOCAL_DETECTOR_DIR / "yolov8n.pt",
]
ANALYSIS_API_BASE_URL = os.getenv("ANALYSIS_API_BASE_URL", "http://127.0.0.1:8000")
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg"}
TEXT_ARTIFACTS = {"pose_sequence", "features", "summary", "kimi_result"}
VIDEO_ARTIFACTS = {"original_video", "original_video_mp4", "pose_video"}
BROWSER_PLAYABLE_VIDEO_EXT = {".mp4"}
SEQ_NO_HEX_RE = re.compile(r"^[0-9a-f]{32}$")
DEFAULT_DOWNLOAD_TIMEOUT_SEC = float(os.getenv("TENNIS_DOWNLOAD_TIMEOUT_SEC", "30"))
DEFAULT_DOWNLOAD_MAX_MB = float(os.getenv("TENNIS_DOWNLOAD_MAX_MB", "2048"))
TASK_LOCK = threading.Lock()

ERROR_CODE_NOT_TENNIS_VIDEO = "NOT_TENNIS_VIDEO"
ERROR_CODE_DOWNLOAD_FAILED = "DOWNLOAD_FAILED"
ERROR_CODE_POSE_FAILED = "POSE_ANALYSIS_FAILED"
ERROR_CODE_FEATURES_FAILED = "FEATURE_EXTRACTION_FAILED"
ERROR_CODE_SUMMARY_FAILED = "SUMMARY_GENERATION_FAILED"
ERROR_CODE_LLM_FAILED = "LLM_ANALYSIS_FAILED"
ERROR_CODE_CLASSIFICATION_FAILED = "CLASSIFICATION_FAILED"
ERROR_CODE_INTERNAL = "INTERNAL_ERROR"
ERROR_CODE_UNKNOWN = "UNKNOWN_ERROR"

WEBHOOK_TIMEOUT_SEC = float(os.getenv("AI_WEBHOOK_TIMEOUT_SEC", "8"))
WEBHOOK_RETRY_TIMES = int(os.getenv("AI_WEBHOOK_RETRY_TIMES", "3"))
WEBHOOK_RETRY_BACKOFF = float(os.getenv("AI_WEBHOOK_RETRY_BACKOFF_SEC", "1.5"))
DEFAULT_WEBHOOK_URL = os.getenv("AI_WEBHOOK_URL", "")
DEFAULT_WEBHOOK_TOKEN = os.getenv("AI_WEBHOOK_TOKEN", "")
WEBHOOK_STAGE_EVENTS = frozenset({"queued", "downloading", "uploaded", "classifying", "rejected", "generating_pose_video", "extracting_features", "summarizing_features", "analyzing_with_kimi", "completed", "failed"})
STEP_LABEL_BY_STAGE = {
    "queued": "queue",
    "downloading": "download",
    "uploaded": "upload",
    "classifying": "analyze",
    "generating_pose_video": "analyze",
    "extracting_features": "analyze",
    "summarizing_features": "analyze",
    "analyzing_with_kimi": "analyze",
    "rejected": "result",
    "completed": "result",
    "failed": "result",
}


def _webhook_timestamp_ms() -> int:
    return int(time.time() * 1000)


def send_ai_analysis_webhook(
    *,
    task_id: str,
    stage: str | None = None,
    result: Any = None,
    step: str | None = None,
    callback_url_override: str | None = None,
    token_override: str | None = None,
) -> tuple[bool, dict[str, Any], str]:
    webhook_url = (callback_url_override or "").strip() or DEFAULT_WEBHOOK_URL
    webhook_token = (token_override or "").strip() or DEFAULT_WEBHOOK_TOKEN
    task = get_task(task_id)
    seq_no = (task.get("seq_no") if isinstance(task, dict) else None) or task_id or ""
    stage_val = (stage or (task.get("stage") if isinstance(task, dict) else None) or "").strip()
    step_val = (step or STEP_LABEL_BY_STAGE.get(stage_val) or "").strip()
    payload_base: dict[str, Any] = {
        "seq_no": seq_no,
        "seqNo": seq_no,
        "stage": stage_val,
        "status": stage_val,
        "step": step_val,
        "result": result if result is not None else (serialize_task(normalize_task_with_derived_artifacts(task), include_text=True) if isinstance(task, dict) else None),
        "timestamp": _webhook_timestamp_ms(),
        "task_id": task_id,
        "taskId": task_id,
    }
    if not webhook_url:
        return False, payload_base, "webhook URL 未配置（AI_WEBHOOK_URL 未设置且 callback_url 为空）"
    parsed = urlparse(webhook_url)
    if parsed.scheme.lower() not in {"http", "https"}:
        return False, payload_base, f"webhook URL 协议不支持: {webhook_url}"
    last_err = ""
    attempt = 0
    for attempt in range(1, WEBHOOK_RETRY_TIMES + 1):
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
        }
        if webhook_token:
            headers["X-AI-Token"] = webhook_token
        try:
            body = json.dumps(payload_base, ensure_ascii=False, default=str).encode("utf-8")
            resp = requests.post(
                webhook_url,
                data=body,
                headers=headers,
                timeout=(3, WEBHOOK_TIMEOUT_SEC),
            )
            if 200 <= resp.status_code < 300:
                return True, payload_base, f"HTTP {resp.status_code}"
            last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as exc:  # pragma: no cover - network side
            last_err = f"{exc.__class__.__name__}: {exc}"
        if attempt < WEBHOOK_RETRY_TIMES:
            time.sleep(min(WEBHOOK_RETRY_BACKOFF * (2 ** (attempt - 1)), 15.0))
    log_with_task(
        get_task_logger(task_id),
        task_id,
        logging.WARNING,
        "webhook_send_failed",
        stage=stage_val or None,
        attempts=attempt,
        last_error=last_err,
        webhook_url=webhook_url,
    )
    return False, payload_base, last_err


def fire_stage_webhook(
    task_id: str,
    stage: str,
    *,
    result: Any = None,
) -> None:
    if not stage or stage not in WEBHOOK_STAGE_EVENTS:
        return
    try:
        task = get_task(task_id)
        url_override = task.get("callback_url") if isinstance(task, dict) else None
        token_override = task.get("callback_token") if isinstance(task, dict) else None
        ok, payload, hint = send_ai_analysis_webhook(
            task_id=task_id,
            stage=stage,
            result=result,
            callback_url_override=url_override if isinstance(url_override, str) else None,
            token_override=token_override if isinstance(token_override, str) else None,
        )
        log_with_task(
            get_task_logger(task_id),
            task_id,
            logging.INFO if ok else logging.WARNING,
            "webhook_sent",
            stage=stage,
            ok=ok,
            hint=hint[:200] if isinstance(hint, str) else "",
        )
    except Exception as exc:  # pragma: no cover - best effort
        log_with_task(
            get_task_logger(task_id),
            task_id,
            logging.WARNING,
            "webhook_fire_exception",
            stage=stage,
            error=f"{exc.__class__.__name__}: {exc}",
        )


def finalize_error(stage: str | None, exc: BaseException | None, message: str) -> tuple[str, str]:
    if stage == "classifying" or exc is not None and exc.__class__.__name__ == "ClassificationFailure":
        code = ERROR_CODE_CLASSIFICATION_FAILED
    else:
        code = ERROR_CODE_UNKNOWN
    exc_name = exc.__class__.__name__ if exc else ""
    if message:
        if "下载" in message or "video_url" in message or (exc_name and "Download" in exc_name):
            code = ERROR_CODE_DOWNLOAD_FAILED
        elif exc_name in ("SubprocessError", "CalledProcessError", "PoseVideoError"):
            if stage == "generating_pose_video" or (message and ("姿态" in message or "pose" in message.lower())):
                code = ERROR_CODE_POSE_FAILED
            elif stage == "extracting_features" or (message and ("特征" in message)):
                code = ERROR_CODE_FEATURES_FAILED
            elif stage == "summarizing_features" or (message and ("摘要" in message)):
                code = ERROR_CODE_SUMMARY_FAILED
        elif (message and ("kimi" in message.lower() or "大模型" in message or "LLM" in message)) or stage == "analyzing_with_kimi":
            code = ERROR_CODE_LLM_FAILED
    if message and ("不是网球" in message or "网球击打" in message):
        code = ERROR_CODE_NOT_TENNIS_VIDEO
    return code, message

TASK_STATE_FILENAME = "task.json"
POSE_FRAME_STEP = int(os.getenv("POSE_FRAME_STEP", "4"))
POSE_MAX_FRAMES = int(os.getenv("POSE_MAX_FRAMES", "100"))


def resolve_default_detector_path() -> str:
    for candidate in DEFAULT_LOCAL_YOLO_CANDIDATES:
        if candidate.is_file():
            return str(candidate)
    return str(DEFAULT_LOCAL_YOLO_CANDIDATES[0])


RTMPOSE_CONFIG = os.getenv("RTMPOSE_CONFIG", str(DEFAULT_LOCAL_RTMPOSE_CONFIG))
RTMPOSE_CHECKPOINT = os.getenv("RTMPOSE_CHECKPOINT", str(DEFAULT_LOCAL_RTMPOSE_CHECKPOINT))
DETECTOR_ID = os.getenv("PERSON_DETECTOR_ID", resolve_default_detector_path())
DETECTION_THRESHOLD = float(os.getenv("DETECTION_THRESHOLD", "0.3"))
YOLO_DETECTION_THRESHOLD = float(os.getenv("YOLO_DETECTION_THRESHOLD", "0.1"))
HOG_DETECTION_THRESHOLD = float(os.getenv("HOG_DETECTION_THRESHOLD", "0.0"))
DEVICE = os.getenv("MODEL_DEVICE", "cpu")

app_state: dict[str, Any] = {"tasks": {}}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_device() -> str:
    if DEVICE == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return DEVICE


def has_local_model_files(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json"))


def has_yolo_weights(path: str) -> bool:
    return os.path.isfile(path) and path.endswith((".pt", ".onnx", ".engine"))


def ensure_rtmpose_runtime() -> None:
    if init_model is None or inference_topdown is None:
        raise RuntimeError("当前环境缺少 RTMPose 运行依赖，请先安装 mmpose、mmengine、mmcv 和 mmdet。")


def build_rtmpose_model(config_path: str, checkpoint_path: str, device: str) -> Any:
    ensure_rtmpose_runtime()
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"未找到 RTMPose 配置文件: {config_path}")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"未找到 RTMPose 权重文件: {checkpoint_path}")
    return init_model(config_path, checkpoint_path, device=device)


def build_hog_detector() -> Any | None:
    if cv2 is None:
        return None
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog


def build_yolo_detector(model_path: str) -> Any | None:
    if YOLO is None or not has_yolo_weights(model_path):
        return None
    return YOLO(model_path)


def resolve_yolo_device(device: str) -> str:
    if device == "cuda":
        return "0"
    if device.startswith("cuda:"):
        return device.split(":", 1)[1]
    return "cpu"


def detect_people_with_yolo(image: Image.Image, yolo_model: Any, device: str) -> np.ndarray:
    results = yolo_model.predict(
        source=np.array(image),
        classes=[0],
        conf=YOLO_DETECTION_THRESHOLD,
        iou=0.4,
        verbose=False,
        device=resolve_yolo_device(device),
    )
    if not results:
        return np.empty((0, 4), dtype=np.float32)

    boxes = results[0].boxes
    if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
        return np.empty((0, 4), dtype=np.float32)

    return boxes.xyxy.detach().cpu().numpy().astype(np.float32)


def detect_people_with_hog(image: Image.Image, hog_detector: Any) -> np.ndarray:
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    rects, weights = hog_detector.detectMultiScale(
        image_bgr,
        winStride=(4, 4),
        padding=(8, 8),
        scale=1.03,
    )
    if len(rects) == 0:
        return np.empty((0, 4), dtype=np.float32)

    boxes_xyxy: list[list[float]] = []
    scores: list[float] = []
    for (x_coord, y_coord, width, height), score in zip(rects, weights.flatten()):
        if float(score) < HOG_DETECTION_THRESHOLD:
            continue
        boxes_xyxy.append(
            [
                float(x_coord),
                float(y_coord),
                float(x_coord + width),
                float(y_coord + height),
            ]
        )
        scores.append(float(score))

    if not boxes_xyxy:
        return np.empty((0, 4), dtype=np.float32)

    keep = nms(
        torch.tensor(boxes_xyxy, dtype=torch.float32),
        torch.tensor(scores, dtype=torch.float32),
        0.4,
    )
    return np.array([boxes_xyxy[index] for index in keep.tolist()], dtype=np.float32)


def prepare_pose_image(image: Image.Image) -> np.ndarray:
    image_rgb = np.array(image)
    if cv2 is not None:
        return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image_rgb[:, :, ::-1].copy()


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def extract_keypoints_from_sample(pose_sample: Any) -> tuple[np.ndarray, np.ndarray]:
    pred_instances = getattr(pose_sample, "pred_instances", None)
    if pred_instances is None or not hasattr(pred_instances, "keypoints"):
        raise RuntimeError("RTMPose 输出中缺少 pred_instances.keypoints")

    keypoints = to_numpy(pred_instances.keypoints)
    scores = getattr(pred_instances, "keypoint_scores", None)
    if scores is None:
        scores = getattr(pred_instances, "keypoints_visible", None)

    scores_array = np.ones((keypoints.shape[-2],), dtype=np.float32) if scores is None else to_numpy(scores)
    if keypoints.ndim == 3:
        keypoints = keypoints[0]
    if scores_array.ndim == 2:
        scores_array = scores_array[0]
    return keypoints, scores_array


def load_models() -> None:
    t0 = time.perf_counter()
    app_logger.info("startup loading models requested_device=%s", DEVICE)
    device = resolve_device()
    t_pose = time.perf_counter()
    pose_model = build_rtmpose_model(RTMPOSE_CONFIG, RTMPOSE_CHECKPOINT, device)
    pose_ms = (time.perf_counter() - t_pose) * 1000
    app_logger.info("startup rtmpose pose_model loaded device=%s pose_load_ms=%.0fms", device, pose_ms)

    detector_enabled = False
    detector_processor = None
    detector_model = None
    detector_backend = None
    t_det = time.perf_counter()
    if has_local_model_files(DETECTOR_ID):
        try:
            from transformers import AutoProcessor, RTDetrForObjectDetection
        except ImportError as exc:
            raise RuntimeError("检测器配置指向 RT-DETR，但当前环境缺少 transformers 依赖。") from exc
        detector_enabled = True
        detector_processor = AutoProcessor.from_pretrained(DETECTOR_ID)
        detector_model = RTDetrForObjectDetection.from_pretrained(DETECTOR_ID).to(device).eval()
        detector_backend = "rtdetr"
    elif has_yolo_weights(DETECTOR_ID):
        yolo_detector = build_yolo_detector(DETECTOR_ID)
        if yolo_detector is not None:
            detector_enabled = True
            detector_model = yolo_detector
            detector_backend = "yolo"
    else:
        hog_detector = build_hog_detector()
        if hog_detector is not None:
            detector_enabled = True
            detector_model = hog_detector
            detector_backend = "hog"
    detector_ms = (time.perf_counter() - t_det) * 1000
    app_logger.info(
        "startup detector loaded backend=%s enabled=%s detector_load_ms=%.0fms",
        detector_backend,
        detector_enabled,
        detector_ms,
    )

    ANALYSIS_TASKS_DIR.mkdir(parents=True, exist_ok=True)
    app_state.update(
        {
            "device": device,
            "detector_enabled": detector_enabled,
            "detector_backend": detector_backend,
            "detector_processor": detector_processor,
            "detector_model": detector_model,
            "pose_model": pose_model,
        }
    )
    total_ms = (time.perf_counter() - t0) * 1000
    app_logger.info(
        "startup complete device=%s detector_backend=%s total_load_ms=%.0fms tasks_dir=%s",
        device,
        detector_backend,
        total_ms,
        str(ANALYSIS_TASKS_DIR),
    )


def ensure_models_loaded() -> None:
    if app_state.get("pose_model") is None:
        load_models()


def unload_models() -> None:
    app_state.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def predict_image(image: Image.Image) -> dict[str, Any]:
    ensure_models_loaded()

    device = app_state["device"]
    detector_enabled = app_state["detector_enabled"]
    detector_backend = app_state["detector_backend"]

    if detector_enabled:
        detector_model = app_state["detector_model"]
        if detector_backend == "rtdetr":
            detector_processor = app_state["detector_processor"]
            detector_inputs = detector_processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                detector_outputs = detector_model(**detector_inputs)
            detection = detector_processor.post_process_object_detection(
                detector_outputs,
                target_sizes=torch.tensor([(image.height, image.width)]),
                threshold=DETECTION_THRESHOLD,
            )[0]
            person_boxes_xyxy = detection["boxes"][detection["labels"] == 0].cpu().numpy()
        elif detector_backend == "yolo":
            person_boxes_xyxy = detect_people_with_yolo(image, detector_model, device)
        else:
            person_boxes_xyxy = detect_people_with_hog(image, detector_model)

        if len(person_boxes_xyxy) == 0:
            return {"num_people": 0, "people": [], "message": "没有检测到人体"}
    else:
        person_boxes_xyxy = np.array([[0.0, 0.0, float(image.width), float(image.height)]], dtype=np.float32)

    pose_results = inference_topdown(
        app_state["pose_model"],
        prepare_pose_image(image),
        person_boxes_xyxy.astype(np.float32),
        bbox_format="xyxy",
    )

    people = []
    for box_xyxy, pose_sample in zip(person_boxes_xyxy.tolist(), pose_results):
        keypoints, scores = extract_keypoints_from_sample(pose_sample)
        people.append(
            {
                "bbox_xyxy": [round(float(v), 3) for v in box_xyxy],
                "keypoints": [[round(float(x), 3), round(float(y), 3)] for x, y in keypoints.tolist()],
                "scores": [round(float(score), 4) for score in scores.tolist()],
            }
        )

    return {
        "num_people": len(people),
        "people": people,
        "pose_backend": "rtmpose",
        "detector_enabled": detector_enabled,
        "detector_backend": detector_backend,
    }


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_models()
    yield
    unload_models()


app = FastAPI(title="RTMPose Tennis Analysis API", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if (FRONTEND_DIST_DIR / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST_DIR / "assets")), name="assets")


def get_task(task_id: str) -> dict[str, Any]:
    with TASK_LOCK:
        task = app_state["tasks"].get(task_id)
    if task is not None:
        return normalize_task_with_derived_artifacts(task)

    recovered = load_task_from_disk(task_id)
    if recovered is not None:
        with TASK_LOCK:
            app_state["tasks"][task_id] = recovered
        return normalize_task_with_derived_artifacts(recovered)

    raise HTTPException(status_code=404, detail="任务不存在")


def set_task(task_id: str, **updates: Any) -> dict[str, Any]:
    with TASK_LOCK:
        task = app_state["tasks"][task_id]
        prev_stage = task.get("stage")
        prev_progress = float(task.get("progress") or 0.0)
        task.update(updates)
        task["updated_at"] = now_iso()
        next_stage = task.get("stage")
        next_progress = float(task.get("progress") or 0.0)
        snapshot = task.copy()
    persist_task(snapshot)
    stage_changed = prev_stage != next_stage
    if stage_changed or ("message" in updates) or abs(next_progress - prev_progress) > 0.01:
        try:
            log_with_task(
                get_task_logger(task_id),
                task_id,
                logging.INFO,
                "task_stage_update",
                stage=next_stage,
                progress=f"{next_progress:.2f}",
                prev_stage=prev_stage,
                message=updates.get("message") if isinstance(updates.get("message"), str) else "",
            )
        except Exception:
            pass
    if stage_changed and next_stage in WEBHOOK_STAGE_EVENTS:
        try:
            fire_stage_webhook(task_id, next_stage)
        except Exception:  # pragma: no cover - best effort hook
            pass
    return snapshot


def task_state_path(task_dir: Path) -> Path:
    return task_dir / TASK_STATE_FILENAME


def normalize_task(task: dict[str, Any]) -> dict[str, Any]:
    task.setdefault("seq_no", task.get("task_id") or "")
    task.setdefault("video_url", "")
    task.setdefault("history_records", [])
    task.setdefault("filename", "")
    task.setdefault("message", "任务状态未知")
    task.setdefault("progress", 0.0)
    task.setdefault("created_at", now_iso())
    task.setdefault("updated_at", task["created_at"])
    task.setdefault("is_tennis_video", None)
    task.setdefault("classification_reason", "")
    task.setdefault("rejection_reason", "")
    task.setdefault("error_code", "")
    task.setdefault("error_message", "")
    task.setdefault("callback_url", "")
    task.setdefault("callback_token", "")
    task.setdefault("artifacts", {})
    for key in ("original_video", "original_video_mp4", "pose_video", "pose_sequence", "features", "summary", "kimi_result"):
        task["artifacts"].setdefault(key, "")
    return task


def persist_task(task: dict[str, Any]) -> None:
    task_dir = Path(task["task_dir"])
    task_dir.mkdir(parents=True, exist_ok=True)
    task = normalize_task(task.copy())
    task_state_path(task_dir).write_text(
        json.dumps(task, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def recover_task_from_dir(task_id: str, task_dir: Path) -> dict[str, Any] | None:
    if not task_dir.is_dir():
        return None

    artifacts: dict[str, str] = {
        "original_video": "",
        "original_video_mp4": "",
        "pose_video": "",
        "pose_sequence": "",
        "features": "",
        "summary": "",
        "kimi_result": "",
    }
    for candidate in task_dir.iterdir():
        if candidate.is_dir():
            continue
        if candidate.name == "original.mp4":
            artifacts["original_video"] = str(candidate)
            artifacts["original_video_mp4"] = str(candidate)
        elif candidate.name.startswith("original.") and candidate.suffix.lower() != ".mp4":
            artifacts["original_video"] = str(candidate)
        elif candidate.name == "original_playback.mp4":
            artifacts["original_video_mp4"] = str(candidate)
        elif candidate.name == "pose_visualized.mp4":
            artifacts["pose_video"] = str(candidate)
        elif candidate.name == "pose_sequence.txt":
            artifacts["pose_sequence"] = str(candidate)
        elif candidate.name == "pose_features.txt":
            artifacts["features"] = str(candidate)
        elif candidate.name == "pose_summary.txt":
            artifacts["summary"] = str(candidate)
        elif candidate.name == "kimi_analysis.txt":
            artifacts["kimi_result"] = str(candidate)

    if artifacts["original_video"] and not artifacts["original_video_mp4"]:
        suffix = Path(artifacts["original_video"]).suffix.lower()
        if suffix in BROWSER_PLAYABLE_VIDEO_EXT:
            artifacts["original_video_mp4"] = artifacts["original_video"]

    if not any(artifacts.values()):
        return None

    created_ts = datetime.fromtimestamp(task_dir.stat().st_ctime, tz=timezone.utc).isoformat()
    task = {
        "task_id": task_id,
        "filename": Path(artifacts["original_video"]).name if artifacts["original_video"] else "",
        "task_dir": str(task_dir),
        "stage": "failed",
        "message": "历史任务状态已恢复，但原始进度信息丢失。请重新分析以获取完整状态。",
        "progress": 1.0,
        "created_at": created_ts,
        "updated_at": now_iso(),
        "is_tennis_video": None,
        "classification_reason": "",
        "rejection_reason": "",
        "error_code": "",
        "error_message": "",
        "callback_url": "",
        "callback_token": "",
        "artifacts": artifacts,
    }
    persist_task(task)
    return task


def load_task_from_disk(task_id: str) -> dict[str, Any] | None:
    task_dir = ANALYSIS_TASKS_DIR / task_id
    state_file = task_state_path(task_dir)
    if state_file.exists():
        try:
            task = json.loads(state_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return recover_task_from_dir(task_id, task_dir)
        return normalize_task_with_derived_artifacts(normalize_task(task))
    return recover_task_from_dir(task_id, task_dir)


def normalize_task_with_derived_artifacts(task: dict[str, Any]) -> dict[str, Any]:
    artifacts = task.setdefault("artifacts", {})
    artifacts.setdefault("original_video_mp4", "")
    ov = artifacts.get("original_video") or ""
    ovmp4 = artifacts.get("original_video_mp4") or ""
    if ov and not ovmp4:
        ov_path = Path(ov)
        if ov_path.suffix.lower() in BROWSER_PLAYABLE_VIDEO_EXT and ov_path.exists():
            artifacts["original_video_mp4"] = ov
        else:
            playback = ov_path.with_name("original_playback.mp4")
            if playback.exists():
                artifacts["original_video_mp4"] = str(playback)
    if not artifacts.get("original_video_mp4") and ov and Path(ov).suffix.lower() not in BROWSER_PLAYABLE_VIDEO_EXT:
        task_dir = Path(task.get("task_dir") or str(Path(ov).parent))
        playback = task_dir / "original_playback.mp4"
        if playback.exists():
            artifacts["original_video_mp4"] = str(playback)
    return task


def _ffmpeg_transcode_to_h264_mp4(src: Path, dst: Path) -> bool:
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return False
    tmp_path = dst.with_suffix(".ffmpeg_tmp.mp4")
    try:
        result = subprocess.run(
            [
                ffmpeg_bin,
                "-y",
                "-i",
                str(src),
                "-vcodec",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-preset",
                "medium",
                "-crf",
                "23",
                "-an",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or not tmp_path.exists() or tmp_path.stat().st_size == 0:
            return False
        tmp_path.replace(dst)
        return True
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        return False


def transcode_original_to_playable_mp4(task_id: str) -> str | None:
    t0 = time.perf_counter()
    task = normalize_task_with_derived_artifacts(get_task(task_id))
    artifacts = task["artifacts"]
    original_video = artifacts.get("original_video") or ""
    if not original_video:
        log_with_task(get_task_logger(task_id), task_id, logging.WARNING, "transcode_missing_original_video")
        return None
    src = Path(original_video)
    if not src.exists():
        log_with_task(get_task_logger(task_id), task_id, logging.WARNING, "transcode_original_video_not_exist path=%s", str(src))
        return None
    if src.suffix.lower() in BROWSER_PLAYABLE_VIDEO_EXT:
        artifacts["original_video_mp4"] = str(src)
        set_task(task_id, artifacts={**artifacts})
        return str(src)
    task_dir = Path(task["task_dir"])
    dst = task_dir / "original_playback.mp4"
    if dst.exists() and dst.stat().st_size > 0:
        artifacts["original_video_mp4"] = str(dst)
        set_task(task_id, artifacts={**artifacts})
        log_with_task(
            get_task_logger(task_id),
            task_id,
            logging.INFO,
            "transcode_reuse_existing_playback",
            stage="uploaded",
            size_bytes=dst.stat().st_size,
            duration_ms=(time.perf_counter() - t0) * 1000,
        )
        return str(dst)
    log_with_task(
        get_task_logger(task_id),
        task_id,
        logging.INFO,
        "transcode_start src_suffix=%s",
        src.suffix.lower(),
        stage="uploaded",
        src_size_bytes=src.stat().st_size if src.exists() else 0,
    )

    ffmpeg_ok = _ffmpeg_transcode_to_h264_mp4(src, dst)
    if ffmpeg_ok:
        out_size = dst.stat().st_size
        artifacts["original_video_mp4"] = str(dst)
        set_task(task_id, artifacts={**artifacts})
        total_ms = (time.perf_counter() - t0) * 1000
        log_with_task(
            get_task_logger(task_id),
            task_id,
            logging.INFO,
            "transcode_complete_ffmpeg",
            stage="uploaded",
            duration_ms=total_ms,
            size_bytes=out_size,
        )
        return str(dst)

    if cv2 is None:
        log_with_task(get_task_logger(task_id), task_id, logging.ERROR, "transcode_ffmpeg_failed_and_cv2_missing")
        return None
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        log_with_task(get_task_logger(task_id), task_id, logging.ERROR, "transcode_capture_open_failed")
        return None
    try:
        fps_value = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        n_frames_est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_width <= 0 or frame_height <= 0:
            log_with_task(
                get_task_logger(task_id),
                task_id,
                logging.ERROR,
                "transcode_invalid_resolution w=%s h=%s",
                frame_width,
                frame_height,
            )
            return None
        fps = float(fps_value) if fps_value and fps_value > 0 else 25.0
        raw_dst = task_dir / "original_playback.mp4v_tmp.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(raw_dst), fourcc, fps, (frame_width, frame_height))
        if not writer.isOpened():
            log_with_task(get_task_logger(task_id), task_id, logging.ERROR, "transcode_writer_open_failed")
            return None
        frames_written = 0
        t_frames = time.perf_counter()
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if frame.shape[0] != frame_height or frame.shape[1] != frame_width:
                    frame = cv2.resize(frame, (frame_width, frame_height))
                writer.write(frame)
                frames_written += 1
        finally:
            writer.release()
    finally:
        cap.release()
    frames_ms = (time.perf_counter() - t_frames) * 1000
    if not raw_dst.exists() or raw_dst.stat().st_size == 0:
        try:
            raw_dst.unlink()
        except FileNotFoundError:
            pass
        log_with_task(
            get_task_logger(task_id),
            task_id,
            logging.ERROR,
            "transcode_failed_empty_output",
            duration_ms=(time.perf_counter() - t0) * 1000,
            frames_written=frames_written,
        )
        return None

    cv2_transcoded_ok = _ffmpeg_transcode_to_h264_mp4(raw_dst, dst)
    try:
        if raw_dst.exists():
            raw_dst.unlink()
    except Exception:
        pass
    if not cv2_transcoded_ok:
        try:
            if dst.exists():
                dst.unlink()
        except Exception:
            pass
        log_with_task(
            get_task_logger(task_id),
            task_id,
            logging.WARNING,
            "transcode_fallback_mp4v_unplayable",
            duration_ms=(time.perf_counter() - t0) * 1000,
            frames_written=frames_written,
        )
        return None

    out_size = dst.stat().st_size
    artifacts["original_video_mp4"] = str(dst)
    set_task(task_id, artifacts={**artifacts})
    total_ms = (time.perf_counter() - t0) * 1000
    log_with_task(
        get_task_logger(task_id),
        task_id,
        logging.INFO,
        "transcode_complete",
        stage="uploaded",
        duration_ms=total_ms,
        frames_written=frames_written,
        frames_est=n_frames_est,
        width=frame_width,
        height=frame_height,
        fps=f"{fps:.2f}",
        size_bytes=out_size,
        frames_process_ms=frames_ms,
    )
    return str(dst)


def async_transcode_original_to_mp4(task_id: str) -> None:
    try:
        transcode_original_to_playable_mp4(task_id)
    except Exception as exc:  # pragma: no cover - unexpected
        log_with_task(
            get_task_logger(task_id),
            task_id,
            logging.ERROR,
            "transcode_uncaught_exception error=%s",
            str(exc),
        )
        return None


def build_artifact_urls(task_id: str, artifacts: dict[str, str | None]) -> dict[str, str]:
    urls = {}
    for kind, value in artifacts.items():
        if value:
            urls[kind] = f"/api/analysis/tasks/{task_id}/artifact?kind={kind}"
    return urls


def serialize_task(task: dict[str, Any], include_text: bool = False) -> dict[str, Any]:
    task_id = task["task_id"]
    summary_text = ""
    kimi_text = ""
    summary_json = None
    kimi_json = None
    if include_text:
        summary_path = task["artifacts"].get("summary")
        kimi_path = task["artifacts"].get("kimi_result")
        if summary_path and Path(summary_path).exists():
            summary_text = Path(summary_path).read_text(encoding="utf-8")
            summary_json = parse_json_from_text(summary_text)
        if kimi_path and Path(kimi_path).exists():
            kimi_text = Path(kimi_path).read_text(encoding="utf-8")
            kimi_json = parse_json_from_text(kimi_text)
        if isinstance(summary_json, dict) and isinstance(kimi_json, dict):
            kimi_json = normalize_analysis_payload(kimi_json)
            derived_scoring = derive_rule_based_score(summary_json, kimi_json)
            kimi_json["overview"]["score"] = derived_scoring["score"]
            kimi_json["scoring"] = derived_scoring
            kimi_text = json.dumps(kimi_json, ensure_ascii=False, indent=2)

    return {
        "taskId": task_id,
        "task_id": task_id,
        "seqNo": task.get("seq_no") or task_id,
        "seq_no": task.get("seq_no") or task_id,
        "stage": task["stage"],
        "message": task["message"],
        "isTennisVideo": task.get("is_tennis_video"),
        "rejectionReason": task.get("rejection_reason"),
        "error": {
            "code": task.get("error_code") or "",
            "message": task.get("error_message") or "",
        } if task.get("stage") in {"failed", "rejected"} else None,
        "progress": task["progress"],
        "createdAt": task["created_at"],
        "updatedAt": task["updated_at"],
        "artifacts": {
            **task["artifacts"],
            "urls": build_artifact_urls(task_id, task["artifacts"]),
        },
        "result": {
            "kimiAnalysis": kimi_text,
            "kimiStructured": kimi_json,
            "summaryText": summary_text,
            "summaryStructured": summary_json,
            "classificationReason": task.get("classification_reason"),
        },
    }


def create_analysis_task(
    filename: str,
    *,
    seq_no: str | None = None,
    history_records: list[dict[str, Any]] | None = None,
    video_url: str | None = None,
) -> dict[str, Any]:
    if seq_no:
        seq_no_normalized = seq_no.strip().lower()
        if not SEQ_NO_HEX_RE.match(seq_no_normalized):
            raise ValueError("seq_no 必须是 32 位无横线 UUID（0-9a-f）")
        task_id = seq_no_normalized
    else:
        task_id = uuid.uuid4().hex
    task_dir = ANALYSIS_TASKS_DIR / task_id
    if task_dir.exists() and any(task_dir.iterdir()):
        raise ValueError(f"seq_no 已存在或任务目录不为空: {task_id}")
    task_dir.mkdir(parents=True, exist_ok=True)
    task = {
        "task_id": task_id,
        "seq_no": task_id,
        "filename": filename,
        "task_dir": str(task_dir),
        "stage": "queued",
        "message": "任务已创建，等待分析",
        "progress": 0.02,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "is_tennis_video": None,
        "classification_reason": "",
        "rejection_reason": "",
        "error_code": "",
        "error_message": "",
        "video_url": video_url or "",
        "history_records": history_records or [],
        "callback_url": "",
        "callback_token": "",
        "artifacts": {
            "original_video": "",
            "original_video_mp4": "",
            "pose_video": "",
            "pose_sequence": "",
            "features": "",
            "summary": "",
            "kimi_result": "",
        },
    }
    with TASK_LOCK:
        app_state["tasks"][task_id] = task
    persist_task(task)
    return task


def extract_video_suffix_from_url(url: str, content_type: str | None = None) -> str:
    parsed = urlparse(url)
    path_name = Path(parsed.path or "").name or "downloaded_video"
    suffix = Path(path_name).suffix.lower()
    if suffix in ALLOWED_VIDEO_EXTENSIONS:
        return suffix
    if content_type:
        ct = content_type.split(";")[0].strip().lower()
        mapping = {
            "video/mp4": ".mp4",
            "video/quicktime": ".mov",
            "video/x-msvideo": ".avi",
            "video/x-matroska": ".mkv",
            "video/webm": ".webm",
            "video/mpeg": ".mpeg",
            "video/mpg": ".mpg",
        }
        if ct in mapping:
            return mapping[ct]
    return ".mp4"


def download_video_from_url(url: str, dest: Path, *, task_id: str) -> tuple[int, str]:
    parsed = urlparse(url)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("video_url 仅支持 http / https 协议")
    task_logger = get_task_logger(task_id)
    size_limit_bytes = int(DEFAULT_DOWNLOAD_MAX_MB * 1024 * 1024)
    downloaded = 0
    content_type = ""
    with requests.get(
        url,
        stream=True,
        timeout=(5, DEFAULT_DOWNLOAD_TIMEOUT_SEC),
        allow_redirects=True,
    ) as resp:
        if resp.status_code >= 400:
            raise RuntimeError(
                f"下载 video_url 失败: HTTP {resp.status_code} {resp.reason or ''}".strip()
            )
        content_type = resp.headers.get("Content-Type", "")
        content_length_raw = resp.headers.get("Content-Length")
        try:
            content_length = int(content_length_raw) if content_length_raw else None
        except ValueError:
            content_length = None
        if content_length is not None and content_length > size_limit_bytes:
            raise RuntimeError(
                f"video_url 文件大小 {content_length} 字节超过上限 {size_limit_bytes} 字节"
            )
        with dest.open("wb") as fp:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                downloaded += len(chunk)
                if downloaded > size_limit_bytes:
                    raise RuntimeError(
                        f"下载 video_url 超过上限 {size_limit_bytes} 字节"
                    )
                fp.write(chunk)
    log_with_task(
        task_logger,
        task_id,
        logging.INFO,
        "url_download_complete",
        stage="uploaded",
        size_bytes=downloaded,
        content_type=content_type,
        path=str(dest),
    )
    return downloaded, content_type


def _is_json_content_type(content_type: str | None) -> bool:
    if not content_type:
        return False
    ct = content_type.split(";")[0].strip().lower()
    return ct in {"application/json", "text/json"}


def run_command(command: list[str], description: str) -> None:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("LANG", "C.UTF-8")
    env.setdefault("LC_ALL", "C.UTF-8")
    try:
        subprocess.run(
            command,
            cwd=str(BASE_DIR),
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else ""
        stdout = exc.stdout.strip() if exc.stdout else ""
        detail = stderr or stdout or f"{description} 执行失败"
        raise RuntimeError(detail) from exc


def run_analysis_pipeline(task_id: str) -> None:
    pipeline_t0 = time.perf_counter()
    task_logger = get_task_logger(task_id)
    task = get_task(task_id)
    task_dir = Path(task["task_dir"])
    original_video_path = Path(task["artifacts"]["original_video"])
    pose_video_path = task_dir / "pose_visualized.mp4"
    pose_sequence_path = task_dir / "pose_sequence.txt"
    features_path = task_dir / "pose_features.txt"
    summary_path = task_dir / "pose_summary.txt"
    kimi_result_path = task_dir / "kimi_analysis.txt"

    try:
        log_with_task(task_logger, task_id, logging.INFO, "pipeline_stage_entry", stage="classifying")
        set_task(task_id, stage="classifying", progress=0.12, message="正在用多模态大模型判断是否为网球击打视频")
        t_cls = time.perf_counter()
        classification = classify_tennis_video(video_path=original_video_path)
        cls_ms = (time.perf_counter() - t_cls) * 1000
        log_with_task(
            task_logger,
            task_id,
            logging.INFO,
            "classification_result",
            stage="classifying",
            is_tennis_video=classification.get("is_tennis_video"),
            duration_ms=cls_ms,
            reason=(classification.get("reason") or "")[:80],
        )
        set_task(
            task_id,
            is_tennis_video=classification["is_tennis_video"],
            classification_reason=classification["reason"],
        )

        if not classification["is_tennis_video"]:
            log_with_task(
                task_logger,
                task_id,
                logging.WARNING,
                "pipeline_rejected_not_tennis",
                stage="rejected",
                duration_ms=(time.perf_counter() - pipeline_t0) * 1000,
            )
            set_task(
                task_id,
                stage="rejected",
                progress=1.0,
                rejection_reason="判断不是网球视频，无法进行分析",
                message="判断不是网球视频，无法进行分析",
                error_code=ERROR_CODE_NOT_TENNIS_VIDEO,
                error_message="判断不是网球视频，无法进行分析",
            )
            return

        set_task(
            task_id,
            stage="generating_pose_video",
            progress=0.3,
            message=f"正在生成姿态可视化视频与关键点序列（每 {POSE_FRAME_STEP} 帧处理一次，最多处理 {POSE_MAX_FRAMES} 帧）",
        )
        t_pose = time.perf_counter()
        run_command(
            [
                sys.executable,
                str(BASE_DIR / "visualize_pose_video.py"),
                str(original_video_path),
                "--output",
                str(pose_video_path),
                "--txt-output",
                str(pose_sequence_path),
                "--url",
                f"{ANALYSIS_API_BASE_URL}/predict",
                "--predict-mode",
                "local",
                "--frame-step",
                str(POSE_FRAME_STEP),
                "--max-frames",
                str(POSE_MAX_FRAMES),
            ],
            "姿态视频生成",
        )
        pose_ms = (time.perf_counter() - t_pose) * 1000
        log_with_task(
            task_logger,
            task_id,
            logging.INFO,
            "pose_video_generated",
            stage="generating_pose_video",
            duration_ms=pose_ms,
            pose_video_size_bytes=pose_video_path.stat().st_size if pose_video_path.exists() else 0,
            pose_sequence_size_bytes=pose_sequence_path.stat().st_size if pose_sequence_path.exists() else 0,
        )
        task = set_task(
            task_id,
            artifacts={
                **task["artifacts"],
                "pose_video": str(pose_video_path),
                "pose_sequence": str(pose_sequence_path),
            },
        )

        set_task(task_id, stage="extracting_features", progress=0.52, message="正在提取网球动作特征")
        t_feat = time.perf_counter()
        run_command(
            [
                sys.executable,
                str(BASE_DIR / "extract_pose_features.py"),
                str(pose_sequence_path),
                "--output",
                str(features_path),
            ],
            "特征提取",
        )
        feat_ms = (time.perf_counter() - t_feat) * 1000
        log_with_task(
            task_logger,
            task_id,
            logging.INFO,
            "features_extracted",
            stage="extracting_features",
            duration_ms=feat_ms,
            features_size_bytes=features_path.stat().st_size if features_path.exists() else 0,
        )
        task = set_task(
            task_id,
            artifacts={
                **task["artifacts"],
                "features": str(features_path),
            },
        )

        set_task(task_id, stage="summarizing_features", progress=0.68, message="正在生成姿态摘要")
        t_sum = time.perf_counter()
        run_command(
            [
                sys.executable,
                str(BASE_DIR / "summarize_pose_features.py"),
                str(features_path),
                "--output",
                str(summary_path),
            ],
            "摘要生成",
        )
        sum_ms = (time.perf_counter() - t_sum) * 1000
        log_with_task(
            task_logger,
            task_id,
            logging.INFO,
            "pose_summary_generated",
            stage="summarizing_features",
            duration_ms=sum_ms,
            summary_size_bytes=summary_path.stat().st_size if summary_path.exists() else 0,
        )
        task = set_task(
            task_id,
            artifacts={
                **task["artifacts"],
                "summary": str(summary_path),
            },
        )

        set_task(task_id, stage="analyzing_with_kimi", progress=0.86, message="正在调用多模态大模型生成网球动作分析")
        t_kimi = time.perf_counter()
        kimi_analysis = analyze_tennis_video(
            video_path=original_video_path,
            pose_video_path=pose_video_path,
            pose_seq_path=pose_sequence_path,
            features_path=features_path,
            summary_path=summary_path,
        )
        kimi_ms = (time.perf_counter() - t_kimi) * 1000
        kimi_result_path.write_text(kimi_analysis, encoding="utf-8")
        log_with_task(
            task_logger,
            task_id,
            logging.INFO,
            "kimi_analysis_complete",
            stage="analyzing_with_kimi",
            duration_ms=kimi_ms,
            kimi_result_size_bytes=kimi_result_path.stat().st_size,
        )
        total_ms = (time.perf_counter() - pipeline_t0) * 1000
        task = set_task(
            task_id,
            artifacts={
                **task["artifacts"],
                "kimi_result": str(kimi_result_path),
            },
            stage="completed",
            progress=1.0,
            message="分析完成",
        )
        log_with_task(
            task_logger,
            task_id,
            logging.INFO,
            "pipeline_complete",
            stage="completed",
            duration_ms=total_ms,
            classifying_ms=cls_ms,
            pose_ms=pose_ms,
            feature_ms=feat_ms,
            summary_ms=sum_ms,
            kimi_ms=kimi_ms,
        )
    except Exception as exc:  # pragma: no cover - background worker
        error_message = str(exc) or exc.__class__.__name__
        total_ms = (time.perf_counter() - pipeline_t0) * 1000
        task = get_task(task_id)
        current_stage = task.get("stage") if isinstance(task, dict) else None
        code, msg = finalize_error(current_stage, exc, error_message)
        log_with_task(
            task_logger,
            task_id,
            logging.ERROR,
            "pipeline_failed",
            stage="failed",
            duration_ms=total_ms,
            error_type=exc.__class__.__name__,
            error_message=error_message,
            error_code=code,
        )
        set_task(
            task_id,
            stage="failed",
            progress=1.0,
            message=f"分析失败: {error_message}",
            error_code=code,
            error_message=msg,
        )


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "device": app_state.get("device"),
        "pose_backend": "rtmpose",
        "pose_config": RTMPOSE_CONFIG,
        "pose_checkpoint": RTMPOSE_CHECKPOINT,
        "detector_id": DETECTOR_ID,
        "detector_enabled": app_state.get("detector_enabled"),
        "detector_backend": app_state.get("detector_backend"),
        "analysis_api_base_url": ANALYSIS_API_BASE_URL,
        "pose_frame_step": POSE_FRAME_STEP,
        "pose_max_frames": POSE_MAX_FRAMES,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="仅支持图片文件")

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:  # pragma: no cover - defensive branch
        raise HTTPException(status_code=400, detail=f"图片解析失败: {exc}") from exc
    return predict_image(image)


def _enqueue_transcode_if_needed(task_id: str, suffix: str) -> None:
    if suffix in BROWSER_PLAYABLE_VIDEO_EXT:
        return
    log_with_task(
        get_task_logger(task_id),
        task_id,
        logging.INFO,
        "transcode_original_enqueue suffix=%s",
        suffix,
        stage="uploaded",
    )
    transcode_thread = threading.Thread(
        target=async_transcode_original_to_mp4,
        args=(task_id,),
        daemon=True,
    )
    transcode_thread.start()


def _start_pipeline_worker(task_id: str) -> None:
    worker = threading.Thread(target=run_analysis_pipeline, args=(task_id,), daemon=True)
    worker.start()
    log_with_task(
        get_task_logger(task_id),
        task_id,
        logging.INFO,
        "pipeline_started",
        stage="classifying",
    )


def _save_original_video_and_update_task(
    *,
    task_id: str,
    task_artifacts: dict[str, Any],
    original_video_path: Path,
    suffix: str,
    message: str,
) -> None:
    direct_ext = suffix in BROWSER_PLAYABLE_VIDEO_EXT
    set_task(
        task_id,
        artifacts={
            **task_artifacts,
            "original_video": str(original_video_path),
            **({"original_video_mp4": str(original_video_path)} if direct_ext else {}),
        },
        message=message,
        progress=0.08,
    )


def _create_task_from_upload(
    *,
    file: UploadFile,
    seq_no: str | None = None,
    history_records: list[dict[str, Any]] | None = None,
    callback_url: str | None = None,
    callback_token: str | None = None,
) -> dict[str, Any]:
    t_start = time.perf_counter()
    suffix = Path(file.filename or "upload.bin").suffix.lower()
    if suffix not in ALLOWED_VIDEO_EXTENSIONS and not (file.content_type or "").startswith("video/"):
        raise HTTPException(status_code=400, detail="仅支持常见视频文件上传")
    try:
        task = create_analysis_task(
            file.filename or "uploaded_video",
            seq_no=seq_no,
            history_records=history_records,
            video_url=None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(callback_url, str) and callback_url.strip():
        task.setdefault("callback_url", callback_url.strip())
    if isinstance(callback_token, str) and callback_token.strip():
        task.setdefault("callback_token", callback_token.strip())
    task = set_task(task["task_id"], callback_url=task.get("callback_url") or "", callback_token=task.get("callback_token") or "")
    task_dir = Path(task["task_dir"])
    target_name = f"original{suffix or '.mp4'}"
    original_video_path = task_dir / target_name
    task_id = task["task_id"]
    app_logger.info(
        "upload create_task task_id=%s original_name=%s suffix=%s content_type=%s",
        task_id,
        file.filename,
        suffix,
        file.content_type,
    )
    with original_video_path.open("wb") as fp:
        shutil.copyfileobj(file.file, fp)
    size_bytes = original_video_path.stat().st_size
    upload_ms = (time.perf_counter() - t_start) * 1000
    log_with_task(
        get_task_logger(task_id),
        task_id,
        logging.INFO,
        "upload_save_complete",
        stage="uploaded",
        duration_ms=upload_ms,
        size_bytes=size_bytes,
        path=str(original_video_path),
    )
    _save_original_video_and_update_task(
        task_id=task_id,
        task_artifacts=task["artifacts"],
        original_video_path=original_video_path,
        suffix=suffix,
        message="视频上传完成，准备开始分析",
    )
    _enqueue_transcode_if_needed(task_id, suffix)
    _start_pipeline_worker(task_id)
    return serialize_task(normalize_task_with_derived_artifacts(get_task(task_id)))


def _create_task_from_video_url(
    *,
    video_url: str,
    seq_no: str | None = None,
    history_records: list[dict[str, Any]] | None = None,
    callback_url: str | None = None,
    callback_token: str | None = None,
) -> dict[str, Any]:
    if not isinstance(video_url, str) or not video_url.strip():
        raise HTTPException(status_code=400, detail="video_url 不能为空")
    parsed = urlparse(video_url.strip())
    if parsed.scheme.lower() not in {"http", "https"}:
        raise HTTPException(status_code=400, detail="video_url 仅支持 http / https 协议")
    suggested_name = Path(parsed.path or "downloaded_video").name or "downloaded_video"
    try:
        task = create_analysis_task(
            suggested_name,
            seq_no=seq_no,
            history_records=history_records,
            video_url=video_url.strip(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(callback_url, str) and callback_url.strip():
        task.setdefault("callback_url", callback_url.strip())
    if isinstance(callback_token, str) and callback_token.strip():
        task.setdefault("callback_token", callback_token.strip())
    task = set_task(task["task_id"], callback_url=task.get("callback_url") or "", callback_token=task.get("callback_token") or "")
    task_dir = Path(task["task_dir"])
    task_id = task["task_id"]
    set_task(
        task_id,
        stage="downloading",
        message="正在下载视频文件",
        progress=0.04,
    )

    def _download_and_run() -> None:
        task_logger = get_task_logger(task_id)
        try:
            suffix_pre = extract_video_suffix_from_url(video_url.strip(), None)
            temp_name = f"original{suffix_pre}"
            original_video_path = task_dir / temp_name
            try:
                _, content_type = download_video_from_url(
                    video_url.strip(), original_video_path, task_id=task_id
                )
            except Exception as exc:  # pragma: no cover - defensive download branch
                err = str(exc) or exc.__class__.__name__
                code = ERROR_CODE_DOWNLOAD_FAILED if any(
                    k in err for k in ("下载", "HTTP", "超时", "timed out", "timeout", "连接", "Connection", "URL", "超过上限")
                ) else ERROR_CODE_INTERNAL
                log_with_task(
                    task_logger,
                    task_id,
                    logging.ERROR,
                    "url_download_failed",
                    stage="downloading",
                    error=err,
                    error_code=code,
                )
                set_task(
                    task_id,
                    stage="failed",
                    progress=1.0,
                    message=f"视频下载失败: {err}",
                    error_code=code,
                    error_message=err,
                )
                return
            suffix = extract_video_suffix_from_url(video_url.strip(), content_type)
            if Path(original_video_path).suffix.lower() != suffix:
                renamed = original_video_path.with_name(f"original{suffix}")
                original_video_path.replace(renamed)
                original_video_path = renamed
            size_bytes = original_video_path.stat().st_size if original_video_path.exists() else 0
            _save_original_video_and_update_task(
                task_id=task_id,
                task_artifacts=task["artifacts"],
                original_video_path=original_video_path,
                suffix=suffix,
                message="视频下载完成，准备开始分析",
            )
            log_with_task(
                task_logger,
                task_id,
                logging.INFO,
                "url_download_saved",
                stage="uploaded",
                size_bytes=size_bytes,
                path=str(original_video_path),
            )
            _enqueue_transcode_if_needed(task_id, suffix)
            _start_pipeline_worker(task_id)
        except Exception as exc:  # pragma: no cover - defensive background thread
            error_message = str(exc) or exc.__class__.__name__
            code, msg = finalize_error(None, exc, error_message)
            log_with_task(
                get_task_logger(task_id),
                task_id,
                logging.ERROR,
                "url_worker_uncaught_exception",
                error_message=error_message,
                error_code=code,
            )
            set_task(
                task_id,
                stage="failed",
                progress=1.0,
                message=f"分析失败: {error_message}",
                error_code=code,
                error_message=msg,
            )

    worker = threading.Thread(target=_download_and_run, daemon=True)
    worker.start()
    return serialize_task(normalize_task_with_derived_artifacts(get_task(task_id)))


@app.post("/api/analysis/tasks")
async def create_task(
    request: Request,
    file: UploadFile | None = File(default=None),
    payload: Any | None = Body(default=None),
) -> dict[str, Any]:
    content_type = request.headers.get("content-type") or ""
    is_json = _is_json_content_type(content_type)
    if not is_json and file is not None:
        form = await request.form()
        seq_no_val: str | None = None
        callback_url_val: str | None = None
        callback_token_val: str | None = None
        for k in ("seq_no", "seqNo"):
            v = form.get(k)
            if isinstance(v, str) and v.strip():
                seq_no_val = v.strip()
                break
        for k in ("callback_url", "callbackUrl", "webhook_url", "webhookUrl"):
            v = form.get(k)
            if isinstance(v, str) and v.strip():
                callback_url_val = v.strip()
                break
        for k in ("callback_token", "callbackToken", "webhook_token", "webhookToken", "ai_token", "aiToken"):
            v = form.get(k)
            if isinstance(v, str) and v.strip():
                callback_token_val = v.strip()
                break
        return _create_task_from_upload(
            file=file,
            seq_no=seq_no_val,
            history_records=None,
            callback_url=callback_url_val,
            callback_token=callback_token_val,
        )
    if is_json:
        try:
            raw_body = await request.body()
            if raw_body:
                decoded = raw_body.decode("utf-8")
                parsed = json.loads(decoded)
                if isinstance(parsed, dict):
                    body = parsed
                elif payload is None:
                    raise HTTPException(status_code=400, detail="请求体必须是 JSON 对象")
                else:
                    body = {}
            elif isinstance(payload, dict):
                body = payload
            elif payload is not None:
                raise HTTPException(status_code=400, detail="请求体必须是 JSON 对象")
            else:
                body = {}
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"请求体 JSON 解析失败: {exc.msg}") from exc
        except Exception as exc:  # pragma: no cover - defensive
            if isinstance(payload, dict):
                body = payload
            else:
                raise HTTPException(status_code=400, detail=f"读取 JSON 请求体失败: {exc}") from exc
        seq_no_val: str | None = body.get("seq_no") or body.get("seqNo")
        video_url_val: str | None = body.get("video_url") or body.get("videoUrl")
        callback_url_val: str | None = (
            body.get("callback_url") or body.get("callbackUrl") or body.get("webhook_url") or body.get("webhookUrl")
        )
        callback_token_val: str | None = (
            body.get("callback_token")
            or body.get("callbackToken")
            or body.get("webhook_token")
            or body.get("webhookToken")
            or body.get("ai_token")
            or body.get("aiToken")
        )
        header_ai_token = request.headers.get("x-ai-token") or request.headers.get("X-AI-Token")
        if isinstance(header_ai_token, str) and header_ai_token.strip() and not callback_token_val:
            callback_token_val = header_ai_token.strip()
        history_raw = body.get("history_records") or body.get("historyRecords")
        history_records: list[dict[str, Any]] | None = None
        if isinstance(history_raw, list):
            if len(history_raw) > 3:
                raise HTTPException(status_code=400, detail="history_records 最多 3 条")
            history_records = [item for item in history_raw if isinstance(item, dict)]
        if isinstance(video_url_val, str) and video_url_val.strip():
            return _create_task_from_video_url(
                video_url=video_url_val,
                seq_no=seq_no_val,
                history_records=history_records,
                callback_url=callback_url_val if isinstance(callback_url_val, str) else None,
                callback_token=callback_token_val if isinstance(callback_token_val, str) else None,
            )
        raise HTTPException(status_code=400, detail="JSON 请求体必须包含 video_url 字段")
    raise HTTPException(
        status_code=400,
        detail="请求格式不支持：请使用 multipart/form-data 上传 file，或使用 application/json 传入 video_url。",
    )


@app.get("/api/analysis/tasks/{task_id}")
def get_task_status(task_id: str, include_text: bool = Query(False)) -> dict[str, Any]:
    return serialize_task(get_task(task_id), include_text=include_text)


@app.get("/api/analysis/tasks/{task_id}/artifact")
def get_task_artifact(
    request: Request,
    task_id: str,
    kind: str = Query(...),
) -> Response:
    task = get_task(task_id)
    if kind not in task["artifacts"]:
        raise HTTPException(status_code=400, detail="不支持的产物类型")

    artifact_path = task["artifacts"].get(kind)
    if not artifact_path:
        raise HTTPException(status_code=404, detail="产物不存在")
    path = Path(artifact_path)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="产物不存在")

    media_type = "text/plain; charset=utf-8"
    if kind in VIDEO_ARTIFACTS:
        media_type = "video/mp4" if path.suffix.lower() == ".mp4" else "video/x-msvideo"
    elif kind in {"summary", "features", "pose_sequence", "kimi_result"}:
        media_type = "application/json"

    if kind not in VIDEO_ARTIFACTS:
        return FileResponse(path=path, media_type=media_type, filename=path.name)

    stat = path.stat()
    file_size = stat.st_size
    last_modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")
    etag = f'"{task_id}-{kind}-{file_size:x}-{int(stat.st_mtime):x}"'
    range_header = request.headers.get("range")
    if_none_match = request.headers.get("if-none-match")
    if_modified_since = request.headers.get("if-modified-since")
    # 分段 Range 请求不走 304 缓存校验，避免浏览器用了空 body 导致视频解码失败
    if not range_header and (
        if_none_match == etag or (if_modified_since and if_modified_since == last_modified)
    ):
        return Response(status_code=304, headers={
            "Cache-Control": "public, max-age=31536000",
            "ETag": etag,
            "Last-Modified": last_modified,
        })
    status_code = 200
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(file_size),
        "Content-Type": media_type,
        "Cache-Control": "public, max-age=31536000",
        "ETag": etag,
        "Last-Modified": last_modified,
    }
    content_length = file_size
    start = 0
    end = file_size - 1

    if range_header and range_header.lower().startswith("bytes="):
        range_value = range_header[len("bytes="):]
        if "," in range_value:
            raise HTTPException(status_code=416, detail="不支持多段 Range 请求")
        start_str, sep, end_str = range_value.partition("-")
        try:
            if start_str == "" and sep == "-" and end_str:
                suffix_length = int(end_str)
                if suffix_length <= 0:
                    raise ValueError
                start = max(0, file_size - suffix_length)
                end = file_size - 1
            else:
                start = int(start_str) if start_str else 0
                end = int(end_str) if end_str else file_size - 1
        except ValueError as exc:
            raise HTTPException(status_code=416, detail="Range 格式不合法") from exc
        if start > end or start < 0 or end >= file_size:
            raise HTTPException(status_code=416, detail=f"Range 超出文件大小，文件大小: {file_size}")
        content_length = end - start + 1
        status_code = 206
        headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
        headers["Content-Length"] = str(content_length)

    with path.open("rb") as fp:
        fp.seek(start)
        data = fp.read(content_length)

    return Response(content=data, status_code=status_code, headers=headers)


@app.get("/api/analysis/tasks/{task_id}/frame")
def get_task_frame(
    request: Request,
    task_id: str,
    time_sec: float = Query(..., ge=0.0),
    kind: str = Query("original_video"),
) -> Response:
    task = get_task(task_id)
    if kind not in VIDEO_ARTIFACTS:
        raise HTTPException(status_code=400, detail="仅支持从视频产物中提取帧")

    artifact_path = task["artifacts"].get(kind)
    if not artifact_path or not Path(artifact_path).exists():
        raise HTTPException(status_code=404, detail="视频产物不存在")
    if cv2 is None:
        raise HTTPException(status_code=500, detail="当前环境缺少 OpenCV，无法提取视频帧")

    video_path = Path(artifact_path)
    video_stat = video_path.stat()
    cache_key = f"{task_id}-{kind}-{time_sec:.3f}-{int(video_stat.st_mtime)}-{video_stat.st_size}"
    import hashlib
    etag = '"' + hashlib.sha1(cache_key.encode("utf-8")).hexdigest() + '"'
    last_modified = datetime.fromtimestamp(video_stat.st_mtime, tz=timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")
    if_none_match = request.headers.get("if-none-match")
    if if_none_match == etag:
        return Response(status_code=304, headers={
            "Cache-Control": "public, max-age=31536000, immutable",
            "ETag": etag,
            "Last-Modified": last_modified,
        })

    capture = cv2.VideoCapture(str(artifact_path))
    if not capture.isOpened():
        raise HTTPException(status_code=500, detail="无法打开视频产物")

    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000.0)
        ok, frame = capture.read()
    finally:
        capture.release()

    if not ok or frame is None:
        raise HTTPException(status_code=404, detail="未能读取指定时间点的视频帧")

    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        raise HTTPException(status_code=500, detail="视频帧编码失败")
    return Response(
        content=encoded.tobytes(),
        media_type="image/jpeg",
        headers={
            "Cache-Control": "public, max-age=31536000, immutable",
            "ETag": etag,
            "Last-Modified": last_modified,
        },
    )


@app.get("/favicon.svg")
def serve_favicon() -> FileResponse:
    favicon_path = FRONTEND_DIST_DIR / "favicon.svg"
    if not favicon_path.exists():
        raise HTTPException(status_code=404, detail="favicon 不存在")
    return FileResponse(path=str(favicon_path), media_type="image/svg+xml", filename="favicon.svg")


@app.get("/", response_class=HTMLResponse)
def serve_index() -> HTMLResponse:
    index_path = FRONTEND_DIST_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>前端尚未构建，请先启动 webapp 或执行构建。</h1>")


@app.get("/task/{task_id}", response_class=HTMLResponse)
def serve_task_page(task_id: str) -> HTMLResponse:
    _ = task_id
    return serve_index()


@app.get("/report/{task_id}", response_class=HTMLResponse)
def serve_report_page(task_id: str) -> HTMLResponse:
    _ = task_id
    return serve_index()
