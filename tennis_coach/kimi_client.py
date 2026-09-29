import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import cv2
from openai import OpenAI

from tennis_coach.logging_utils import ensure_logging_configured, get_app_logger

ensure_logging_configured(os.getenv("TENNIS_LOG_LEVEL", "INFO"))
llm_logger = get_app_logger("kimi_client")


DEFAULT_PROVIDER = os.getenv("TENNIS_LLM_PROVIDER", "moonshot").strip().lower() or "moonshot"

PROVIDER_ALIASES = {
    "moonshot": "moonshot",
    "kimi": "moonshot",
    "dashscope": "dashscope",
    "aliyun": "dashscope",
    "bailian": "dashscope",
    "alibaba": "dashscope",
    "token_plan": "token_plan",
    "token-plan": "token_plan",
    "tokenplan": "token_plan",
    "maas": "token_plan",
    "aliyun_maas": "token_plan",
    "aliyun-maas": "token_plan",
    "aliyunmaas": "token_plan",
}

DEFAULT_MODELS = {
    "moonshot": "kimi-k2.6",
    "dashscope": "qwen3-vl-plus",
    "token_plan": "Qwen3.8-Max",
}

MAX_INLINE_VIDEO_BYTES = 18 * 1024 * 1024
MIN_INLINE_VIDEO_DURATION_SEC = 1.5
MAX_FRAME_IMAGES = 6
MAX_IMAGE_SIDE = 960
JPEG_QUALITY = 80


def normalize_provider(provider: str | None = None) -> str:
    raw = (provider or DEFAULT_PROVIDER).strip().lower()
    if not raw:
        return "moonshot"
    if raw in PROVIDER_ALIASES:
        return PROVIDER_ALIASES[raw]

    # Be tolerant of malformed env values like "dashscopeexport" caused by shell paste mistakes.
    for alias in sorted(PROVIDER_ALIASES.keys(), key=len, reverse=True):
        if raw.startswith(alias) or alias in raw:
            return PROVIDER_ALIASES[alias]

    raise EnvironmentError(f"不支持的 TENNIS_LLM_PROVIDER: {raw}")


def default_model_for_provider(provider: str) -> str:
    return DEFAULT_MODELS.get(provider, DEFAULT_MODELS["moonshot"])


def provider_display_name(provider: str) -> str:
    names = {
        "moonshot": "Kimi",
        "dashscope": "阿里百炼",
        "token_plan": "阿里云推理服务(Token Plan)",
    }
    return names.get(provider, provider)


def get_api_key(provider: str | None = None) -> str:
    current_provider = normalize_provider(provider)
    if current_provider == "moonshot":
        key_env = "MOONSHOT_API_KEY"
    elif current_provider == "token_plan":
        key_env = "TOKEN_PLAN_API_KEY"
    else:
        key_env = "DASHSCOPE_API_KEY"
    api_key = os.getenv(key_env, "").strip()
    if not api_key:
        raise EnvironmentError(f"请先设置环境变量 {key_env}")
    try:
        api_key.encode("ascii")
    except UnicodeEncodeError as exc:
        raise EnvironmentError(f"{key_env} 必须是有效的 ASCII 字符串") from exc
    return api_key


def get_base_url(provider: str | None = None) -> str:
    current_provider = normalize_provider(provider)
    if current_provider == "dashscope":
        return os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").strip()
    if current_provider == "token_plan":
        return os.getenv(
            "TOKEN_PLAN_BASE_URL",
            "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        ).strip()
    return os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1").strip()


def resolve_model(model: str | None = None, provider: str | None = None) -> str:
    if model and model.strip():
        return model.strip()
    current_provider = normalize_provider(provider)
    # 允许用全局环境变量覆盖
    global_override = os.getenv("TENNIS_LLM_MODEL", "").strip()
    if global_override:
        return global_override
    # 支持按 provider 单独指定
    provider_model_env = {
        "moonshot": "MOONSHOT_MODEL",
        "dashscope": "DASHSCOPE_MODEL",
        "token_plan": "TOKEN_PLAN_MODEL",
    }.get(current_provider)
    if provider_model_env:
        provider_override = os.getenv(provider_model_env, "").strip()
        if provider_override:
            return provider_override
    return default_model_for_provider(current_provider)


def build_client(api_key: str | None = None, provider: str | None = None) -> OpenAI:
    current_provider = normalize_provider(provider)
    return OpenAI(
        api_key=api_key or get_api_key(current_provider),
        base_url=get_base_url(current_provider),
    )


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def ascii_safe_text(text: str) -> str:
    if not text:
        return ""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text.encode("unicode_escape").decode("ascii")
    return json.dumps(parsed, ensure_ascii=True, indent=2)


def encode_video_as_data_url(video_path: Path) -> str:
    with video_path.open("rb") as fp:
        video_data = fp.read()
    suffix = video_path.suffix.lstrip(".").lower() or "mp4"
    encoded = base64.b64encode(video_data).decode("utf-8")
    return f"data:video/{suffix};base64,{encoded}"


def encode_frame_as_data_url(frame: Any) -> str:
    if frame is None:
        raise ValueError("视频帧为空，无法编码")

    height, width = frame.shape[:2]
    longest_side = max(height, width)
    if longest_side > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / float(longest_side)
        frame = cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)

    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise RuntimeError("视频帧 JPEG 编码失败")
    payload = base64.b64encode(encoded.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{payload}"


def get_video_duration_sec(video_path: Path) -> float:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return 0.0
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    finally:
        capture.release()
    if fps <= 0.0 or frame_count <= 0.0:
        return 0.0
    return frame_count / fps


def select_frame_timestamps(video_path: Path, summary_payload: dict[str, Any] | None = None, max_frames: int = MAX_FRAME_IMAGES) -> list[float]:
    references = collect_reference_timestamps(summary_payload or {})
    if references:
        ordered = sorted({round(float(item), 3) for item in references if isinstance(item, (int, float)) and float(item) >= 0.0})
        if len(ordered) <= max_frames:
            return ordered
        if max_frames == 1:
            return [ordered[len(ordered) // 2]]
        sampled: list[float] = []
        last_index = len(ordered) - 1
        for step in range(max_frames):
            index = round(step * last_index / (max_frames - 1))
            sampled.append(ordered[index])
        return sampled

    duration_sec = get_video_duration_sec(video_path)
    if duration_sec <= 0.0:
        return [0.0]

    if max_frames <= 1:
        return [round(duration_sec / 2.0, 3)]

    start = min(0.25, duration_sec * 0.15)
    end = max(duration_sec - 0.25, start)
    if end <= start:
        end = duration_sec
    if end <= 0.0:
        return [0.0]

    sampled = []
    for step in range(max_frames):
        ratio = step / (max_frames - 1)
        sampled.append(round(start + (end - start) * ratio, 3))
    return sampled


def extract_frame_content_items(
    video_path: Path,
    *,
    label: str,
    summary_payload: dict[str, Any] | None = None,
    max_frames: int = MAX_FRAME_IMAGES,
    reason: str = "当前改为发送关键帧序列",
) -> list[dict[str, Any]]:
    timestamps = select_frame_timestamps(video_path, summary_payload=summary_payload, max_frames=max_frames)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频文件: {video_path}")

    content_items: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": f"{label}{reason}，以下发送 {len(timestamps)} 张代表性关键帧，请结合这些画面按时间顺序理解动作。",
        }
    ]
    try:
        for timestamp_sec in timestamps:
            capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_sec) * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            content_items.append(
                {
                    "type": "text",
                    "text": f"{label}关键帧，时间点约 {timestamp_sec:.2f}s",
                }
            )
            content_items.append(
                {
                    "type": "image_url",
                    "image_url": {"url": encode_frame_as_data_url(frame)},
                }
            )
    finally:
        capture.release()

    if len(content_items) == 1:
        raise RuntimeError(f"未能从视频中提取可用关键帧: {video_path}")
    return content_items


def build_video_content_items(
    video_path: Path,
    *,
    label: str,
    provider: str | None = None,
    summary_payload: dict[str, Any] | None = None,
    max_inline_video_bytes: int = MAX_INLINE_VIDEO_BYTES,
    min_inline_video_duration_sec: float = MIN_INLINE_VIDEO_DURATION_SEC,
    force_frames: bool = False,
) -> list[dict[str, Any]]:
    current_provider = normalize_provider(provider)
    file_size = video_path.stat().st_size if video_path.exists() else 0
    duration_sec = get_video_duration_sec(video_path)
    if force_frames:
        return extract_frame_content_items(video_path, label=label, summary_payload=summary_payload)
    if file_size and file_size > max_inline_video_bytes:
        return extract_frame_content_items(video_path, label=label, summary_payload=summary_payload, reason="过大，当前改为发送关键帧序列")
    if current_provider in {"dashscope", "token_plan"} and duration_sec > 0.0 and duration_sec < min_inline_video_duration_sec:
        return extract_frame_content_items(video_path, label=label, summary_payload=summary_payload, reason="过短，当前改为发送关键帧序列")
    if file_size:
        return [{"type": "video_url", "video_url": {"url": encode_video_as_data_url(video_path)}}]
    return extract_frame_content_items(video_path, label=label, summary_payload=summary_payload)


def is_too_short_video_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "video file is too short" in message or "the video modality input does not meet the requirements" in message


def parse_json_from_text(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    candidates = [stripped]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(stripped[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _to_text_list(value: Any, fallback: list[str] | None = None, limit: int | None = None) -> list[str]:
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
    elif value is None:
        items = []
    else:
        text = str(value).strip()
        items = [text] if text else []
    if not items and fallback:
        items = fallback
    return items[:limit] if limit is not None else items


def _to_status(value: Any, allowed: set[str], fallback: str) -> str:
    status = str(value).strip().lower()
    return status if status in allowed else fallback


def collect_reference_timestamps(summary_payload: dict[str, Any]) -> list[float]:
    timestamps: set[float] = set()

    peak_windows = summary_payload.get("peak_windows")
    if isinstance(peak_windows, dict):
        for windows in peak_windows.values():
            if not isinstance(windows, list):
                continue
            for window in windows:
                if not isinstance(window, dict):
                    continue
                peak_time = window.get("peak_timestamp_sec")
                if isinstance(peak_time, (int, float)):
                    timestamps.add(round(float(peak_time), 4))
                frame_window = window.get("window")
                if isinstance(frame_window, list):
                    for frame in frame_window:
                        if not isinstance(frame, dict):
                            continue
                        time_value = frame.get("timestamp_sec")
                        if isinstance(time_value, (int, float)):
                            timestamps.add(round(float(time_value), 4))

    key_frames = summary_payload.get("coordination_metrics", {}).get("key_frames")
    if isinstance(key_frames, dict):
        for item in key_frames.values():
            if not isinstance(item, dict):
                continue
            time_value = item.get("timestamp_sec")
            if isinstance(time_value, (int, float)):
                timestamps.add(round(float(time_value), 4))

    phases = summary_payload.get("action_phase_segments")
    if isinstance(phases, list):
        for phase in phases:
            if not isinstance(phase, dict):
                continue
            for key in ("start_time_sec", "end_time_sec"):
                time_value = phase.get(key)
                if isinstance(time_value, (int, float)):
                    timestamps.add(round(float(time_value), 4))

    return sorted(timestamps)


def nearest_timestamp(value: Any, references: list[float]) -> float:
    try:
        target = float(value)
    except (TypeError, ValueError):
        return references[0] if references else 0.0
    if not references:
        return round(target, 4)
    return min(references, key=lambda item: abs(item - target))


def phase_for_timestamp(timestamp_sec: float, summary_payload: dict[str, Any]) -> str | None:
    phases = summary_payload.get("action_phase_segments")
    if not isinstance(phases, list):
        return None

    best_name = None
    best_distance = None
    for phase in phases:
        if not isinstance(phase, dict):
            continue
        start_time = phase.get("start_time_sec")
        end_time = phase.get("end_time_sec")
        if not isinstance(start_time, (int, float)) or not isinstance(end_time, (int, float)):
            continue
        if start_time <= timestamp_sec <= end_time:
            return str(phase.get("phase", "")).strip() or None
        distance = min(abs(timestamp_sec - float(start_time)), abs(timestamp_sec - float(end_time)))
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_name = str(phase.get("phase", "")).strip() or None
    return best_name


def align_analysis_payload_with_summary(parsed: dict[str, Any], summary_payload: dict[str, Any]) -> dict[str, Any]:
    references = collect_reference_timestamps(summary_payload)

    highlights = parsed.get("highlights")
    if isinstance(highlights, dict):
        aligned_time = nearest_timestamp(highlights.get("timestamp_sec"), references)
        highlights["timestamp_sec"] = aligned_time
        aligned_phase = phase_for_timestamp(aligned_time, summary_payload)
        if aligned_phase:
            highlights["phase"] = aligned_phase

    improvements = parsed.get("improvements")
    segments = improvements.get("segments") if isinstance(improvements, dict) else None
    if isinstance(segments, list):
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            aligned_time = nearest_timestamp(segment.get("timestamp_sec"), references)
            segment["timestamp_sec"] = aligned_time
            aligned_phase = phase_for_timestamp(aligned_time, summary_payload)
            if aligned_phase:
                segment["phase"] = aligned_phase

    return parsed


def normalize_analysis_payload(parsed: dict[str, Any]) -> dict[str, Any]:
    overview = parsed.get("overview") or {}
    highlights = parsed.get("highlights") or {}
    improvements = parsed.get("improvements") or {}
    skills_status = parsed.get("skills_status") or {}
    confidence = parsed.get("confidence") or {}

    raw_segments = improvements.get("segments")
    segments: list[dict[str, Any]] = []
    if isinstance(raw_segments, list):
        for item in raw_segments[:4]:
            if not isinstance(item, dict):
                continue
            raw_checkpoints = item.get("checkpoints")
            checkpoints: list[dict[str, str]] = []
            if isinstance(raw_checkpoints, list):
                for checkpoint in raw_checkpoints[:6]:
                    if not isinstance(checkpoint, dict):
                        continue
                    checkpoints.append(
                        {
                            "name": str(checkpoint.get("name", "")).strip() or "动作检查点",
                            "status": _to_status(checkpoint.get("status"), {"good", "bad"}, "bad"),
                            "detail": str(checkpoint.get("detail", "")).strip() or "当前片段缺少更细的动作说明。",
                        }
                    )
            if not checkpoints:
                checkpoints = [
                    {
                        "name": "动作检查点",
                        "status": "bad",
                        "detail": "当前片段缺少更细的动作说明。",
                    }
                ]
            segments.append(
                {
                    "timestamp_sec": float(item.get("timestamp_sec", 0.0) or 0.0),
                    "phase": str(item.get("phase", "")).strip() or "关键动作阶段",
                    "headline": str(item.get("headline", "")).strip() or "这是需要重点关注的动作片段，我们逐帧来看。",
                    "frame_caption": str(item.get("frame_caption", "")).strip() or "滑动查看关键动作序列",
                    "checkpoints": checkpoints,
                }
            )

    if not segments:
        segments = [
            {
                "timestamp_sec": float(highlights.get("timestamp_sec", 0.0) or 0.0),
                "phase": str(highlights.get("phase", "")).strip() or "关键动作阶段",
                "headline": "当前视频可见信息有限，建议先结合原视频继续复核这个关键动作片段。",
                "frame_caption": "滑动查看关键动作序列",
                "checkpoints": [
                    {
                        "name": "动作完整度",
                        "status": "bad",
                        "detail": "当前结构化结果不完整，建议重新生成分析或补充更清晰的视频。",
                    }
                ],
            }
        ]

    raw_items = skills_status.get("items")
    items: list[dict[str, str]] = []
    if isinstance(raw_items, list):
        for item in raw_items[:6]:
            if not isinstance(item, dict):
                continue
            items.append(
                {
                    "name": str(item.get("name", "")).strip() or "专项能力",
                    "status": _to_status(item.get("status"), {"pass", "needs_work"}, "needs_work"),
                    "detail": str(item.get("detail", "")).strip() or "当前能力项缺少更细说明。",
                }
            )
    if not items:
        items = [
            {
                "name": "动作完整度",
                "status": "needs_work",
                "detail": "当前结果未能稳定生成专项达标项，建议重新分析。",
            }
        ]

    try:
        score = int(round(float(overview.get("score", 0) or 0)))
    except (TypeError, ValueError):
        score = 0
    score = max(0, min(100, score))

    try:
        overall_confidence = float(confidence.get("overall_confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        overall_confidence = 0.0
    overall_confidence = max(0.0, min(1.0, overall_confidence))

    return {
        "overview": {
            "score": score,
            "coach_title": str(overview.get("coach_title", "")).strip() or "AI COACH 鼓励建议",
            "coach_advice": str(overview.get("coach_advice", "")).strip() or "当前动作里已经有不错的基础，建议结合视频继续复核并挑一个重点动作先稳下来。",
        },
        "highlights": {
            "section_title": str(highlights.get("section_title", "")).strip() or "技术闪光点",
            "section_label": str(highlights.get("section_label", "")).strip() or "EXCELLENT",
            "headline": str(highlights.get("headline", "")).strip() or "当前动作中存在可继续放大的积极信号。",
            "timestamp_sec": float(highlights.get("timestamp_sec", 0.0) or 0.0),
            "phase": str(highlights.get("phase", "")).strip() or "关键动作阶段",
            "bullets": _to_text_list(highlights.get("bullets"), fallback=["当前片段缺少更多亮点描述。"], limit=4),
        },
        "improvements": {
            "section_title": str(improvements.get("section_title", "")).strip() or "继续提升点",
            "section_label": str(improvements.get("section_label", "")).strip() or "NEXT STEP",
            "segments": segments,
        },
        "skills_status": {
            "section_title": str(skills_status.get("section_title", "")).strip() or "专项达标",
            "items": items,
        },
        "training_plan_cta": str(parsed.get("training_plan_cta", "")).strip() or "生成训练计划",
        "confidence": {
            "overall_confidence": overall_confidence,
            "limitations": _to_text_list(
                confidence.get("limitations"),
                fallback=["当前视频仅基于人体关键点与可见画面判断，未包含球路和拍头速度。"],
                limit=4,
            ),
        },
    }


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _range_score(
    value: float | None,
    *,
    low: float,
    ideal_low: float,
    ideal_high: float,
    high: float,
    default: float = 0.5,
) -> float:
    if value is None:
        return default
    if value <= low or value >= high:
        return 0.0
    if ideal_low <= value <= ideal_high:
        return 1.0
    if value < ideal_low:
        return (value - low) / max(ideal_low - low, 1e-6)
    return (high - value) / max(high - ideal_high, 1e-6)


def _threshold_score(
    value: float | None,
    *,
    low: float,
    target: float,
    default: float = 0.5,
) -> float:
    if value is None:
        return default
    if value <= low:
        return 0.0
    if value >= target:
        return 1.0
    return (value - low) / max(target - low, 1e-6)


def _small_is_good(
    value: float | None,
    *,
    ideal: float,
    max_bad: float,
    default: float = 0.5,
) -> float:
    if value is None:
        return default
    if value <= ideal:
        return 1.0
    if value >= max_bad:
        return 0.0
    return (max_bad - value) / max(max_bad - ideal, 1e-6)


def _extract_feature_stats(summary_payload: dict[str, Any], feature_name: str) -> dict[str, Any]:
    feature_stats = summary_payload.get("feature_stats")
    if not isinstance(feature_stats, dict):
        return {}
    stats = feature_stats.get(feature_name)
    return stats if isinstance(stats, dict) else {}


def _extract_peak_value(summary_payload: dict[str, Any], feature_name: str, *, absolute: bool = False) -> float | None:
    peak_events = summary_payload.get("peak_events")
    if isinstance(peak_events, dict):
        entries = peak_events.get(feature_name)
        if isinstance(entries, list):
            values = []
            for item in entries:
                if not isinstance(item, dict):
                    continue
                value = _safe_float(item.get("value"))
                if value is None:
                    continue
                values.append(abs(value) if absolute else value)
            if values:
                return max(values)

    feature_stats = _extract_feature_stats(summary_payload, feature_name)
    max_blob = feature_stats.get("max") if isinstance(feature_stats, dict) else None
    if isinstance(max_blob, dict):
        value = _safe_float(max_blob.get("value"))
        if value is not None:
            return abs(value) if absolute else value
    return None


def _longest_visibility_segment_sec(summary_payload: dict[str, Any]) -> float | None:
    segments = summary_payload.get("visibility_segments")
    if not isinstance(segments, list):
        return None
    durations = []
    for item in segments:
        if not isinstance(item, dict):
            continue
        duration = _safe_float(item.get("duration_sec"))
        if duration is not None:
            durations.append(duration)
    return max(durations) if durations else None


def _stance_commitment_ratio(summary_payload: dict[str, Any]) -> float | None:
    stance_counts = summary_payload.get("stance_type_distribution") or summary_payload.get("video_summary", {}).get("stance_type_counts")
    if not isinstance(stance_counts, dict):
        return None
    values = [int(v) for v in stance_counts.values() if isinstance(v, (int, float))]
    total = sum(values)
    if total <= 0:
        return None
    return max(values) / total


def _action_phase_count(summary_payload: dict[str, Any]) -> int:
    segments = summary_payload.get("action_phase_segments")
    if not isinstance(segments, list):
        return 0
    return sum(1 for item in segments if isinstance(item, dict))


def derive_rule_based_score(summary_payload: dict[str, Any], analysis_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    video_summary = summary_payload.get("video_summary") if isinstance(summary_payload.get("video_summary"), dict) else {}

    visible_ratio = _safe_float(video_summary.get("person_visible_ratio"))
    stance_ratio = _safe_float(video_summary.get("avg_stance_width_to_hip_width_ratio"))
    torso_angle = _safe_float(video_summary.get("avg_torso_angle_deg"))
    left_knee = _safe_float(video_summary.get("avg_left_knee_angle_deg"))
    right_knee = _safe_float(video_summary.get("avg_right_knee_angle_deg"))
    knee_mean = None if left_knee is None or right_knee is None else (left_knee + right_knee) / 2.0
    knee_diff = _safe_float(video_summary.get("avg_left_right_knee_angle_diff_deg"))

    separation_peak = _extract_peak_value(summary_payload, "shoulder_hip_separation_angle_deg")
    hip_rotation_peak = _extract_peak_value(summary_payload, "hip_rotation_velocity_deg_per_sec", absolute=True)
    shoulder_rotation_peak = _extract_peak_value(summary_payload, "shoulder_rotation_velocity_deg_per_sec", absolute=True)
    dominant_wrist_peak = max(
        filter(
            None,
            [
                _extract_peak_value(summary_payload, "left_wrist_speed", absolute=True),
                _extract_peak_value(summary_payload, "right_wrist_speed", absolute=True),
            ],
        ),
        default=None,
    )
    dominant_ankle_peak = max(
        filter(
            None,
            [
                _extract_peak_value(summary_payload, "left_ankle_speed", absolute=True),
                _extract_peak_value(summary_payload, "right_ankle_speed", absolute=True),
            ],
        ),
        default=None,
    )
    hip_center_peak = _extract_peak_value(summary_payload, "hip_center_speed", absolute=True)
    head_stability_std = _safe_float(_extract_feature_stats(summary_payload, "head_to_mid_hip_distance_px").get("std"))
    longest_visibility_sec = _longest_visibility_segment_sec(summary_payload)
    phase_count = _action_phase_count(summary_payload)
    stance_commitment = _stance_commitment_ratio(summary_payload)

    pose_mechanics = (
        6.0 * _range_score(stance_ratio, low=0.85, ideal_low=1.1, ideal_high=2.2, high=3.2)
        + 4.0 * _range_score(torso_angle, low=25.0, ideal_low=42.0, ideal_high=66.0, high=82.0)
        + 4.0 * _range_score(knee_mean, low=138.0, ideal_low=150.0, ideal_high=172.0, high=179.0)
        + 10.0 * _range_score(separation_peak, low=4.0, ideal_low=20.0, ideal_high=78.0, high=95.0)
        + 7.0 * _threshold_score(hip_rotation_peak, low=40.0, target=260.0)
        + 5.0 * _threshold_score(shoulder_rotation_peak, low=30.0, target=220.0)
        + 2.0 * _small_is_good(head_stability_std, ideal=4.0, max_bad=18.0)
        + 2.0 * _small_is_good(knee_diff, ideal=8.0, max_bad=28.0)
    )

    motion_completeness = (
        10.0 * _range_score(visible_ratio, low=0.12, ideal_low=0.4, ideal_high=0.95, high=1.01)
        + 8.0 * _threshold_score(longest_visibility_sec, low=0.06, target=0.35)
        + 4.0 * _threshold_score(float(phase_count), low=0.0, target=3.0, default=0.0)
        + 4.0 * _range_score(stance_commitment, low=0.15, ideal_low=0.4, ideal_high=0.85, high=1.01)
        + 6.0 * _threshold_score(dominant_wrist_peak, low=150.0, target=1400.0)
        + 4.0 * _threshold_score(dominant_ankle_peak, low=30.0, target=250.0)
        + 4.0 * _threshold_score(hip_center_peak, low=40.0, target=220.0)
    )

    checkpoint_good_ratio = 0.5
    skills_pass_ratio = 0.5
    highlight_bonus = 0.0
    segments_bonus = 0.0
    if isinstance(analysis_payload, dict):
        improvements = analysis_payload.get("improvements")
        if isinstance(improvements, dict):
            segments = improvements.get("segments")
            if isinstance(segments, list):
                checkpoint_statuses = []
                valid_segments = 0
                for segment in segments:
                    if not isinstance(segment, dict):
                        continue
                    valid_segments += 1
                    checkpoints = segment.get("checkpoints")
                    if not isinstance(checkpoints, list):
                        continue
                    for checkpoint in checkpoints:
                        if isinstance(checkpoint, dict):
                            checkpoint_statuses.append(str(checkpoint.get("status", "")).strip())
                if checkpoint_statuses:
                    good_count = sum(1 for status in checkpoint_statuses if status == "good")
                    checkpoint_good_ratio = good_count / len(checkpoint_statuses)
                if 2 <= valid_segments <= 4:
                    segments_bonus = 1.0

        skills = analysis_payload.get("skills_status")
        if isinstance(skills, dict):
            items = skills.get("items")
            if isinstance(items, list):
                statuses = [str(item.get("status", "")).strip() for item in items if isinstance(item, dict)]
                if statuses:
                    pass_count = sum(1 for status in statuses if status == "pass")
                    skills_pass_ratio = pass_count / len(statuses)

        highlights = analysis_payload.get("highlights")
        if isinstance(highlights, dict):
            bullets = highlights.get("bullets")
            timestamp = _safe_float(highlights.get("timestamp_sec"))
            if isinstance(bullets, list) and len(bullets) >= 2 and timestamp is not None and timestamp >= 0.0:
                highlight_bonus = 1.0

    llm_calibration = _clamp(
        10.0
        + (checkpoint_good_ratio - 0.5) * 8.0
        + (skills_pass_ratio - 0.5) * 6.0
        + highlight_bonus
        + segments_bonus,
        0.0,
        20.0,
    )

    raw_total = pose_mechanics + motion_completeness + llm_calibration
    total_score = int(round(_clamp(raw_total, 0.0, 100.0)))
    score = max(45, min(95, total_score))

    return {
        "score": score,
        "raw_total": round(raw_total, 2),
        "components": {
            "pose_mechanics": round(pose_mechanics, 2),
            "motion_completeness": round(motion_completeness, 2),
            "llm_calibration": round(llm_calibration, 2),
        },
        "metrics": {
            "visible_ratio": visible_ratio,
            "longest_visibility_sec": longest_visibility_sec,
            "action_phase_count": phase_count,
            "stance_ratio": stance_ratio,
            "torso_angle": torso_angle,
            "knee_mean": knee_mean,
            "knee_diff": knee_diff,
            "separation_peak": separation_peak,
            "hip_rotation_peak": hip_rotation_peak,
            "shoulder_rotation_peak": shoulder_rotation_peak,
            "dominant_wrist_peak": dominant_wrist_peak,
            "dominant_ankle_peak": dominant_ankle_peak,
            "hip_center_peak": hip_center_peak,
            "head_stability_std": head_stability_std,
            "stance_commitment": stance_commitment,
            "checkpoint_good_ratio": round(checkpoint_good_ratio, 3),
            "skills_pass_ratio": round(skills_pass_ratio, 3),
        },
        "method": "rule_pose_40 + temporal_40 + llm_calibration_20",
    }


def classify_tennis_video(
    video_path: Path,
    model: str | None = None,
    provider: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    current_provider = normalize_provider(provider)
    resolved_model = resolve_model(model, current_provider)
    client = build_client(api_key, current_provider)
    prompt = """
You are a video classification assistant.
Decide whether this video clearly shows a real human performing a tennis hitting, swinging, rally, serving, or tennis training action.

Rules:
1. Return true only if the video clearly contains a human doing a tennis-related striking or swing action.
2. Return false if it is not tennis, too unclear to judge, not a human tennis action, only walking, another sport, or an empty scene.
3. Output strict JSON only. Do not add markdown or extra explanation.
4. The "reason" field must be a short sentence in Simplified Chinese.

JSON schema:
{
  "is_tennis_video": true,
  "reason": "short reason in Simplified Chinese"
}
""".strip()
    force_frames = False
    attempts = 0
    while True:
        attempts += 1
        visual_content = build_video_content_items(
            video_path,
            label="原始视频",
            provider=current_provider,
            force_frames=force_frames,
        )
        visual_mode = "frames" if force_frames or any(it.get("type") == "image_url" for it in visual_content) else "video_inline"
        t_call = time.perf_counter()
        try:
            completion = client.chat.completions.create(
                model=resolved_model,
                messages=[
                    {"role": "system", "content": "You are a strict video classifier that must output valid JSON only."},
                    {
                        "role": "user",
                        "content": [
                            *visual_content,
                            {"type": "text", "text": prompt},
                        ],
                    },
                ],
            )
            call_ms = (time.perf_counter() - t_call) * 1000
            llm_logger.info(
                "[llm_classify] provider=%s model=%s visual_mode=%s force_frames=%s attempt=%s http_ms=%.0fms total_elapsed_ms=%.0fms",
                current_provider,
                resolved_model,
                visual_mode,
                force_frames,
                attempts,
                call_ms,
                (time.perf_counter() - t0) * 1000,
            )
            break
        except Exception as exc:
            call_ms = (time.perf_counter() - t_call) * 1000
            llm_logger.warning(
                "[llm_classify_error] provider=%s model=%s attempt=%s error_type=%s error=%s elapsed_ms=%.0fms",
                current_provider,
                resolved_model,
                attempts,
                exc.__class__.__name__,
                str(exc)[:160],
                call_ms,
            )
            if not force_frames and is_too_short_video_error(exc):
                force_frames = True
                continue
            raise
    content = completion.choices[0].message.content or ""
    parsed = parse_json_from_text(content)
    if parsed:
        is_tennis = bool(parsed.get("is_tennis_video"))
        reason = str(parsed.get("reason", "")).strip() or "模型未提供理由"
    else:
        lowered = content.replace(" ", "")
        is_tennis = "true" in lowered or ("网球" in lowered and "不是" not in lowered)
        reason = content.strip() or "模型返回内容无法解析，已使用兜底规则判断"
    llm_logger.info(
        "[llm_classify_result] provider=%s model=%s is_tennis_video=%s reason=%s response_chars=%s parsed_ok=%s total_ms=%.0fms",
        current_provider,
        resolved_model,
        is_tennis,
        (reason or "")[:80],
        len(content),
        parsed is not None,
        (time.perf_counter() - t0) * 1000,
    )

    return {
        "is_tennis_video": is_tennis,
        "reason": reason,
        "raw_response": content,
        "provider": current_provider,
        "model": resolved_model,
    }


def build_analysis_prompt(
    video_path: Path,
    pose_video_path: Path,
    pose_seq_path: Path,
    features_path: Path,
    summary_path: Path,
    summary_text: str,
    features_text: str,
    pose_seq_text: str,
) -> str:
    summary_safe = ascii_safe_text(summary_text)
    features_safe = ascii_safe_text(features_text)
    pose_seq_safe = ascii_safe_text(pose_seq_text)
    summary_payload = parse_json_from_text(summary_text) or {}
    reference_timestamps = collect_reference_timestamps(summary_payload)
    action_phases = summary_payload.get("action_phase_segments") if isinstance(summary_payload.get("action_phase_segments"), list) else []

    reference_timestamp_block = (
        ", ".join(f"{timestamp:.4f}" for timestamp in reference_timestamps[:80]) if reference_timestamps else "无"
    )
    action_phase_block = json.dumps(action_phases, ensure_ascii=False, indent=2)[:4000] if action_phases else "[]"

    return f"""
You are a professional tennis coach and movement analyst.
Most uploaded tennis videos are from beginners or amateur enthusiasts rather than elite athletes.
Use the uploaded video together with the structured pose-analysis data below to evaluate the athlete's movement quality.

Goals:
1. Identify the most likely stroke type or action phase, such as preparation, loading, acceleration, follow-through, and recovery.
2. Evaluate the kinetic chain, especially hip rotation, shoulder rotation, shoulder-hip separation, center-of-mass movement, lower-body contribution, and upper-body sequencing.
3. Analyze stance, stability, head control, left-right symmetry, and possible compensation patterns.
4. Provide specific and actionable coaching suggestions.
5. Clearly distinguish which conclusions come from direct video observation and which come from pose-derived features.
6. If the pose features and visual impression disagree, explain the likely reason.
7. Keep the coaching tone positive, encouraging, and confidence-building.

Output requirements:
- Output STRICT JSON only. Do not add markdown, preamble, code fences, or explanatory text.
- All string values must be in Simplified Chinese.
- The JSON must follow this schema exactly:
{{
  "overview": {{
    "score": 72,
    "coach_title": "AI COACH 鼓励建议",
    "coach_advice": "一句到两句的总评，先肯定当前动作基础，再给出一个最值得优先提升的方向"
  }},
  "highlights": {{
    "section_title": "技术闪光点",
    "section_label": "EXCELLENT",
    "headline": "一句亮点总结",
    "timestamp_sec": 0.7,
    "phase": "例如：蓄力引拍阶段",
    "bullets": [
      "亮点1",
      "亮点2"
    ]
  }},
  "improvements": {{
    "section_title": "继续提升点",
    "section_label": "NEXT STEP",
    "segments": [
      {{
        "timestamp_sec": 0.7,
        "phase": "蓄力引拍阶段",
        "headline": "这是你0.7s蓄力引拍阶段，我们看看还能怎样更顺一点",
        "frame_caption": "滑动查看关键动作序列",
        "checkpoints": [
          {{
            "name": "非持拍手指向",
            "status": "good",
            "detail": "先给出鼓励性判断，再说明原因和动作影响"
          }},
          {{
            "name": "蹬拍头",
            "status": "good",
            "detail": "先给出鼓励性判断，再说明原因和动作影响"
          }},
          {{
            "name": "击球点",
            "status": "bad",
            "detail": "避免打击式措辞，改成下一步怎么调整会更好 + 原因 + 动作影响"
          }}
        ]
      }}
    ]
  }},
  "skills_status": {{
    "section_title": "专项达标",
    "items": [
      {{
        "name": "腿部转换",
        "status": "pass",
        "detail": "简短达标说明"
      }},
      {{
        "name": "挥拍完整度",
        "status": "pass",
        "detail": "简短达标说明"
      }},
      {{
        "name": "脚步",
        "status": "needs_work",
        "detail": "简短问题说明"
      }},
            {{
        "name": "下肢驱动",
        "status": "needs_work",
        "detail": "简短问题说明"
      }}
    ]
  }},
  "training_plan_cta": "生成训练计划",
  "confidence": {{
    "overall_confidence": 0.74,
    "limitations": [
      "限制1",
      "限制2"
    ]
  }}
}}

Rules for content quality:
- score must be an integer from 0 to 100.
- CRITICAL SCORE RULES (read carefully before writing JSON):
  1. NEVER copy or reuse the example values from the JSON schema above. The example score=72 and overall_confidence=0.74 are for schema reference only, not for actual output.
  2. NEVER output a score that equals a common cluster value such as 82, 80, or 72 unless the video quality truly matches that exact level.
  3. Scores MUST distribute realistically across the 50–95 range for typical amateur or beginner videos.
  4. If two videos differ noticeably in technique quality, their scores MUST differ by at least 4 points.
  5. Use these scoring benchmarks as a guide, and deviate freely according to actual observation:
     - 50–62: Beginner with significant compensations or incomplete movement (not unsafe)
     - 63–72: Solid beginner baseline with 2-3 key elements in place
     - 73–84: Amateur with good kinetic chain foundation and 4+ checkpoints passing
     - 85–95: Advanced amateur with consistent technique, only minor refinements needed
     - 96–100: Reserve for clearly elite or near-competitive technique, rarely used
- Use 1 highlight block and 2 to 4 improvement segments.
- Each improvement segment must contain 4 to 6 checkpoints.
- checkpoint status must be either "good" or "bad".
- skills_status item status must be either "pass" or "needs_work".
- Prefer timestamps that correspond to meaningful action moments visible in the video.
- Do not invent racket/ball facts if they are not visible.
- If information is uncertain, say so in the detail text and also mention it in confidence.limitations.
- Assume the athlete is usually a beginner or amateur player unless the video clearly shows advanced competitive technique.
- Start from what the athlete is already doing reasonably well, then explain the next improvement step.
- Avoid harsh, dismissive, or overly professional criticism.
- Even when something is weak, phrase it as coachable progress, such as "再稳一点会更好", "如果这里再提前一点会更顺", "这个基础已经有了，可以继续加强".
- Do not give unrealistically low scores for ordinary amateur technique. Reserve very low scores for movements that are clearly incomplete or unsafe.
- All timestamps in highlights and improvements.segments MUST come from the provided reference timestamps. Do not invent new timestamps.
- The text, phase, and checkpoints for each segment must match the selected timestamp's visible motion as closely as possible.
- If you are unsure about a moment, choose the nearest valid reference timestamp instead of fabricating a cleaner number like 0.8 or 1.0.
- Phase names should reuse the provided action phases whenever possible.

Important notes:
- The current pose data contains human keypoints only. It does not include racket head speed, ball trajectory, or exact contact point.
- Some frames in the summary may have no detected person because of detection dropouts, not necessarily because the motion stopped.
- Do not just repeat numbers. Combine direct visual evidence from the video with the structured features.

Input files:
- Original video: {video_path}
- Pose visualization video: {pose_video_path}
- Pose sequence file: {pose_seq_path}
- Pose feature file: {features_path}
- Pose summary file: {summary_path}

Pose summary content:
{summary_safe[:16000]}

Reference timestamps that you are allowed to use:
{reference_timestamp_block}

Action phases that you should align with:
{action_phase_block}

Pose feature excerpt (first 6000 chars):
{features_safe[:6000]}

Pose sequence excerpt (first 4000 chars):
{pose_seq_safe[:4000]}
""".strip()


def analyze_tennis_video(
    *,
    video_path: Path,
    pose_video_path: Path,
    pose_seq_path: Path,
    features_path: Path,
    summary_path: Path,
    model: str | None = None,
    provider: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.2,
) -> str:
    t0 = time.perf_counter()
    current_provider = normalize_provider(provider)
    resolved_model = resolve_model(model, current_provider)
    client = build_client(api_key, current_provider)
    summary_text = read_text(summary_path)
    summary_payload = parse_json_from_text(summary_text) or {}
    prompt_text = build_analysis_prompt(
        video_path=video_path,
        pose_video_path=pose_video_path,
        pose_seq_path=pose_seq_path,
        features_path=features_path,
        summary_path=summary_path,
        summary_text=summary_text,
        features_text=read_text(features_path),
        pose_seq_text=read_text(pose_seq_path),
    )
    ref_timestamps_count = len(collect_reference_timestamps(summary_payload))
    llm_logger.info(
        "[llm_analyze_start] provider=%s model=%s summary_chars=%s features_chars=%s pose_seq_chars=%s reference_timestamps_count=%s prompt_chars=%s",
        current_provider,
        resolved_model,
        len(summary_text),
        len(read_text(features_path)),
        len(read_text(pose_seq_path)),
        ref_timestamps_count,
        len(prompt_text),
    )
    original_force_frames = False
    pose_force_frames = False
    attempts = 0
    while True:
        attempts += 1
        original_visual_content = build_video_content_items(
            video_path,
            label="原始视频",
            provider=current_provider,
            summary_payload=summary_payload,
            force_frames=original_force_frames,
        )
        pose_visual_content = build_video_content_items(
            pose_video_path,
            label="姿态可视化视频",
            provider=current_provider,
            summary_payload=summary_payload,
            force_frames=pose_force_frames,
        )
        orig_mode = "frames" if original_force_frames or any(it.get("type") == "image_url" for it in original_visual_content) else "video_inline"
        pose_mode = "frames" if pose_force_frames or any(it.get("type") == "image_url" for it in pose_visual_content) else "video_inline"
        orig_images = sum(1 for it in original_visual_content if it.get("type") == "image_url")
        pose_images = sum(1 for it in pose_visual_content if it.get("type") == "image_url")
        t_call = time.perf_counter()
        try:
            completion = client.chat.completions.create(
                model=resolved_model,
                # temperature=temperature,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a professional tennis coach using {provider_display_name(current_provider)} who explains movement causality carefully and precisely.",
                    },
                    {
                        "role": "user",
                        "content": [
                            *original_visual_content,
                            *pose_visual_content,
                            {"type": "text", "text": prompt_text},
                        ],
                    },
                ],
            )
            call_ms = (time.perf_counter() - t_call) * 1000
            llm_logger.info(
                "[llm_analyze_call] provider=%s model=%s attempt=%s orig_mode=%s pose_mode=%s orig_images=%s pose_images=%s http_ms=%.0fms elapsed_ms=%.0fms",
                current_provider,
                resolved_model,
                attempts,
                orig_mode,
                pose_mode,
                orig_images,
                pose_images,
                call_ms,
                (time.perf_counter() - t0) * 1000,
            )
            break
        except Exception as exc:
            call_ms = (time.perf_counter() - t_call) * 1000
            llm_logger.warning(
                "[llm_analyze_error] provider=%s model=%s attempt=%s error_type=%s error=%s http_ms=%.0fms elapsed_ms=%.0fms",
                current_provider,
                resolved_model,
                attempts,
                exc.__class__.__name__,
                str(exc)[:200],
                call_ms,
                (time.perf_counter() - t0) * 1000,
            )
            if is_too_short_video_error(exc) and (not original_force_frames or not pose_force_frames):
                original_force_frames = True
                pose_force_frames = True
                continue
            raise
    content = completion.choices[0].message.content or ""
    parsed = parse_json_from_text(content)
    parsed_ok = parsed is not None
    aligned_ok = False
    derived_score: int | None = None
    if parsed is not None:
        try:
            aligned = align_analysis_payload_with_summary(parsed, summary_payload)
            normalized = normalize_analysis_payload(aligned)
            derived_scoring = derive_rule_based_score(summary_payload, normalized)
            normalized["overview"]["score"] = derived_scoring["score"]
            normalized["scoring"] = derived_scoring
            derived_score = derived_scoring["score"]
            content = json.dumps(normalized, ensure_ascii=False, indent=2)
            aligned_ok = True
        except Exception as align_exc:
            llm_logger.warning(
                "[llm_analyze_align_error] provider=%s model=%s error_type=%s error=%s",
                current_provider,
                resolved_model,
                align_exc.__class__.__name__,
                str(align_exc)[:160],
            )
    total_ms = (time.perf_counter() - t0) * 1000
    llm_logger.info(
        "[llm_analyze_result] provider=%s model=%s attempts=%s parsed_ok=%s aligned_ok=%s derived_score=%s response_chars=%s total_ms=%.0fms",
        current_provider,
        resolved_model,
        attempts,
        parsed_ok,
        aligned_ok,
        derived_score,
        len(content),
        total_ms,
    )
    return content
