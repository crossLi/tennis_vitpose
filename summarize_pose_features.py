import argparse
import json
import statistics
from pathlib import Path


KEY_FEATURES = [
    "bbox_area",
    "torso_length",
    "torso_angle_deg",
    "shoulder_hip_separation_angle_deg",
    "shoulder_rotation_velocity_deg_per_sec",
    "hip_rotation_velocity_deg_per_sec",
    "shoulder_hip_separation_angular_velocity_deg_per_sec",
    "stance_width",
    "stance_width_to_hip_width_ratio",
    "head_to_mid_hip_distance_px",
    "left_elbow_angle_deg",
    "right_elbow_angle_deg",
    "left_knee_angle_deg",
    "right_knee_angle_deg",
    "left_right_elbow_angle_diff_deg",
    "left_right_knee_angle_diff_deg",
    "center_speed",
    "hip_center_speed",
    "left_wrist_speed",
    "right_wrist_speed",
    "left_ankle_speed",
    "right_ankle_speed",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将关键点特征整理成适合大模型分析的摘要")
    parser.add_argument("input", help="输入特征文件路径")
    parser.add_argument(
        "--output",
        default="",
        help="输出摘要路径，默认与输入同名并追加 _summary.txt",
    )
    return parser.parse_args()


def load_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def present_frames(frames: list[dict]) -> list[dict]:
    return [frame for frame in frames if frame.get("person_present")]


def feature_values(frames: list[dict], feature_name: str) -> list[tuple[int, float, float]]:
    values = []
    for frame in frames:
        if not frame.get("person_present"):
            continue
        feature_map = frame.get("features", {})
        value = feature_map.get(feature_name)
        if isinstance(value, (int, float)):
            values.append((frame["frame_index"], frame["timestamp_sec"], float(value)))
    return values


def calc_stat_triplet(values: list[tuple[int, float, float]]) -> dict | None:
    if not values:
        return None
    numeric = [value for _, _, value in values]
    min_item = min(values, key=lambda item: item[2])
    max_item = max(values, key=lambda item: item[2])
    return {
        "mean": round(statistics.fmean(numeric), 3),
        "std": round(statistics.pstdev(numeric), 3) if len(numeric) > 1 else 0.0,
        "min": {
            "frame_index": min_item[0],
            "timestamp_sec": round(min_item[1], 4),
            "value": round(min_item[2], 3),
        },
        "max": {
            "frame_index": max_item[0],
            "timestamp_sec": round(max_item[1], 4),
            "value": round(max_item[2], 3),
        },
    }


def visibility_segments(frames: list[dict]) -> list[dict]:
    segments = []
    start = None
    previous = None
    for frame in frames:
        if frame.get("person_present"):
            if start is None:
                start = frame
            previous = frame
        elif start is not None and previous is not None:
            segments.append(
                {
                    "start_frame": start["frame_index"],
                    "end_frame": previous["frame_index"],
                    "start_time_sec": round(start["timestamp_sec"], 4),
                    "end_time_sec": round(previous["timestamp_sec"], 4),
                    "duration_sec": round(previous["timestamp_sec"] - start["timestamp_sec"], 4),
                }
            )
            start = None
            previous = None
    if start is not None and previous is not None:
        segments.append(
            {
                "start_frame": start["frame_index"],
                "end_frame": previous["frame_index"],
                "start_time_sec": round(start["timestamp_sec"], 4),
                "end_time_sec": round(previous["timestamp_sec"], 4),
                "duration_sec": round(previous["timestamp_sec"] - start["timestamp_sec"], 4),
            }
        )
    return segments


def top_events(frames: list[dict], feature_name: str, limit: int = 5, by_abs: bool = False) -> list[dict]:
    values = feature_values(frames, feature_name)
    values.sort(key=lambda item: abs(item[2]) if by_abs else item[2], reverse=True)
    return [
        {
            "frame_index": frame_index,
            "timestamp_sec": round(timestamp_sec, 4),
            "value": round(value, 3),
        }
        for frame_index, timestamp_sec, value in values[:limit]
    ]


def stance_distribution(frames: list[dict]) -> dict:
    counts = {}
    for frame in frames:
        stance = frame.get("features", {}).get("stance_type_heuristic")
        if not stance:
            continue
        counts[stance] = counts.get(stance, 0) + 1
    return counts


def dominant_side(payload: dict) -> dict:
    summary = payload.get("summary", {})
    left = summary.get("avg_left_wrist_speed")
    right = summary.get("avg_right_wrist_speed")
    if left is None and right is None:
        side = None
    elif right is None or (left is not None and left > right):
        side = "left"
    else:
        side = "right"
    return {
        "dominant_wrist_speed_side": side,
        "avg_left_wrist_speed": left,
        "avg_right_wrist_speed": right,
    }


def peak_window_for_feature(frames: list[dict], feature_name: str, limit: int = 3, radius: int = 3, by_abs: bool = False):
    scored = []
    for idx, frame in enumerate(frames):
        if not frame.get("person_present"):
            continue
        value = frame.get("features", {}).get(feature_name)
        if not isinstance(value, (int, float)):
            continue
        score = abs(value) if by_abs else value
        scored.append((idx, frame["frame_index"], frame["timestamp_sec"], float(value), score))

    scored.sort(key=lambda item: item[4], reverse=True)
    selected = []
    used_frame_indices = set()
    for idx, frame_index, timestamp_sec, value, _ in scored:
        if any(abs(frame_index - used) <= radius for used in used_frame_indices):
            continue
        used_frame_indices.add(frame_index)
        window = []
        for raw in frames[max(0, idx - radius) : min(len(frames), idx + radius + 1)]:
            item_value = raw.get("features", {}).get(feature_name)
            window.append(
                {
                    "frame_index": raw["frame_index"],
                    "timestamp_sec": round(raw["timestamp_sec"], 4),
                    "person_present": raw.get("person_present", False),
                    "value": round(item_value, 3) if isinstance(item_value, (int, float)) else None,
                }
            )
        selected.append(
            {
                "peak_frame_index": frame_index,
                "peak_timestamp_sec": round(timestamp_sec, 4),
                "peak_value": round(value, 3),
                "window": window,
            }
        )
        if len(selected) >= limit:
            break
    return selected


def feature_peak_frame(frames: list[dict], feature_name: str, by_abs: bool = False):
    events = top_events(frames, feature_name, limit=1, by_abs=by_abs)
    return events[0] if events else None


def head_stability_summary(frames: list[dict]) -> dict:
    xs = feature_values(frames, "nose_offset_x_from_mid_hip_px")
    ys = feature_values(frames, "nose_offset_y_from_mid_hip_px")
    distances = feature_values(frames, "head_to_mid_hip_distance_px")

    def compact_stats(values):
        if not values:
            return None
        numeric = [value for _, _, value in values]
        return {
            "mean": round(statistics.fmean(numeric), 3),
            "std": round(statistics.pstdev(numeric), 3) if len(numeric) > 1 else 0.0,
            "range": round(max(numeric) - min(numeric), 3),
        }

    return {
        "nose_offset_x_from_mid_hip_px": compact_stats(xs),
        "nose_offset_y_from_mid_hip_px": compact_stats(ys),
        "head_to_mid_hip_distance_px": compact_stats(distances),
    }


def symmetry_summary(frames: list[dict]) -> dict:
    summary = {}
    for feature_name in (
        "left_right_elbow_angle_diff_deg",
        "left_right_knee_angle_diff_deg",
        "left_right_arm_length_diff_px",
        "left_right_leg_length_diff_px",
        "wrist_height_diff_px",
        "ankle_height_diff_px",
    ):
        summary[feature_name] = calc_stat_triplet(feature_values(frames, feature_name))
    return summary


def first_settle_frame(frames: list[dict], feature_name: str, start_frame: int, peak_value: float):
    threshold = abs(peak_value) * 0.35
    candidates = [frame for frame in frames if frame["frame_index"] >= start_frame and frame.get("person_present")]
    consecutive = 0
    settle = None
    for frame in candidates:
        value = frame.get("features", {}).get(feature_name)
        if isinstance(value, (int, float)) and abs(value) <= threshold:
            consecutive += 1
            if settle is None:
                settle = frame
            if consecutive >= 2:
                return settle
        else:
            consecutive = 0
            settle = None
    return candidates[-1] if candidates else None


def action_phase_segments(frames: list[dict], segments: list[dict], dominant_wrist_feature: str) -> list[dict]:
    if not segments:
        return []
    longest = max(segments, key=lambda item: item["duration_sec"])
    segment_frames = [
        frame
        for frame in frames
        if frame.get("person_present")
        and longest["start_frame"] <= frame["frame_index"] <= longest["end_frame"]
    ]
    if not segment_frames:
        return []

    sep_peak = feature_peak_frame(segment_frames, "shoulder_hip_separation_angle_deg")
    wrist_peak = feature_peak_frame(segment_frames, dominant_wrist_feature)
    wrist_peak_value = wrist_peak["value"] if wrist_peak else None
    settle = first_settle_frame(segment_frames, dominant_wrist_feature, wrist_peak["frame_index"], wrist_peak_value) if wrist_peak else segment_frames[-1]

    boundaries = []
    start = segment_frames[0]
    boundaries.append(("准备/蓄力", start, None))
    if sep_peak:
        boundaries[-1] = ("准备/蓄力", start, sep_peak)
        if wrist_peak and wrist_peak["frame_index"] > sep_peak["frame_index"]:
            sep_frame = next(frame for frame in segment_frames if frame["frame_index"] == sep_peak["frame_index"])
            wrist_frame = next(frame for frame in segment_frames if frame["frame_index"] == wrist_peak["frame_index"])
            boundaries.append(("加速挥拍", sep_frame, wrist_frame))
    if wrist_peak and settle:
        wrist_frame = next(frame for frame in segment_frames if frame["frame_index"] == wrist_peak["frame_index"])
        boundaries.append(("收拍", wrist_frame, settle))
    if settle and settle["frame_index"] < segment_frames[-1]["frame_index"]:
        boundaries.append(("回位", settle, segment_frames[-1]))

    phases = []
    for name, start_frame, end_frame in boundaries:
        if start_frame is None:
            continue
        if end_frame is None:
            end_frame = segment_frames[-1]
        if end_frame["frame_index"] < start_frame["frame_index"]:
            continue
        phases.append(
            {
                "phase": name,
                "start_frame": start_frame["frame_index"],
                "end_frame": end_frame["frame_index"],
                "start_time_sec": round(start_frame["timestamp_sec"], 4),
                "end_time_sec": round(end_frame["timestamp_sec"], 4),
                "duration_sec": round(end_frame["timestamp_sec"] - start_frame["timestamp_sec"], 4),
            }
        )
    return phases


def coordination_metrics(frames: list[dict], dominant_wrist_feature: str) -> dict:
    hip_peak = feature_peak_frame(frames, "hip_rotation_velocity_deg_per_sec", by_abs=True)
    shoulder_peak = feature_peak_frame(frames, "shoulder_rotation_velocity_deg_per_sec", by_abs=True)
    sep_peak = feature_peak_frame(frames, "shoulder_hip_separation_angle_deg")
    sep_vel_peak = feature_peak_frame(frames, "shoulder_hip_separation_angular_velocity_deg_per_sec", by_abs=True)
    wrist_peak = feature_peak_frame(frames, dominant_wrist_feature)
    center_peak = feature_peak_frame(frames, "center_speed")

    def delta_seconds(first, second):
        if not first or not second:
            return None
        return round(second["timestamp_sec"] - first["timestamp_sec"], 4)

    return {
        "key_frames": {
            "hip_rotation_velocity_peak": hip_peak,
            "shoulder_rotation_velocity_peak": shoulder_peak,
            "shoulder_hip_separation_peak": sep_peak,
            "shoulder_hip_separation_velocity_peak": sep_vel_peak,
            "dominant_wrist_speed_peak": wrist_peak,
            "center_speed_peak": center_peak,
        },
        "time_deltas_sec": {
            "hip_to_shoulder_rotation_peak": delta_seconds(hip_peak, shoulder_peak),
            "shoulder_rotation_to_wrist_speed_peak": delta_seconds(shoulder_peak, wrist_peak),
            "separation_peak_to_wrist_speed_peak": delta_seconds(sep_peak, wrist_peak),
            "separation_velocity_peak_to_wrist_speed_peak": delta_seconds(sep_vel_peak, wrist_peak),
            "hip_rotation_peak_to_center_speed_peak": delta_seconds(hip_peak, center_peak),
        },
    }


def unavailable_without_extra_detectors():
    return [
        "racket_head_speed",
        "racket_trajectory",
        "ball_contact_height",
        "ball_contact_position_relative_to_body",
        "racket_body_coordination",
        "incoming_ball_height_and_timing",
    ]


def build_llm_brief(payload: dict, segments: list[dict], stats: dict, peaks: dict, windows: dict, stance_counts: dict, dominant: dict, coordination: dict, phases: list[dict]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "视频人体姿态摘要（适合直接提供给大模型分析）",
        f"- 源文件: {payload.get('source_file')}",
        f"- 视频参数: fps={payload.get('fps')}, 分辨率={payload.get('frame_width')}x{payload.get('frame_height')}",
        f"- 总帧数: {summary.get('total_frames')}, 有人帧数: {summary.get('person_visible_frames')}, 可见比例: {summary.get('person_visible_ratio')}",
        f"- 可见片段数: {len(segments)}",
        f"- 握拍侧/主导侧启发式: {dominant.get('dominant_wrist_speed_side')}",
        f"- 站位类型分布(启发式): {stance_counts}",
    ]

    if segments:
        longest = max(segments, key=lambda item: item["duration_sec"])
        lines.append(
            f"- 最长连续可见片段: frame {longest['start_frame']} 到 {longest['end_frame']}，"
            f"时间 {longest['start_time_sec']}s 到 {longest['end_time_sec']}s，持续 {longest['duration_sec']}s"
        )

    def stat_line(label: str, key: str):
        stat = stats.get(key)
        if not stat:
            return
        lines.append(
            f"- {label}: 平均 {stat['mean']}, 标准差 {stat['std']}, "
            f"最小 {stat['min']['value']}@frame {stat['min']['frame_index']}, "
            f"最大 {stat['max']['value']}@frame {stat['max']['frame_index']}"
        )

    stat_line("肩髋分离角", "shoulder_hip_separation_angle_deg")
    stat_line("肩部旋转速度", "shoulder_rotation_velocity_deg_per_sec")
    stat_line("髋部旋转速度", "hip_rotation_velocity_deg_per_sec")
    stat_line("肩髋分离角速度", "shoulder_hip_separation_angular_velocity_deg_per_sec")
    stat_line("站位宽度", "stance_width")
    stat_line("左肘角", "left_elbow_angle_deg")
    stat_line("右肘角", "right_elbow_angle_deg")
    stat_line("左膝角", "left_knee_angle_deg")
    stat_line("右膝角", "right_knee_angle_deg")
    stat_line("人体中心速度", "center_speed")
    stat_line("左手腕速度", "left_wrist_speed")
    stat_line("右手腕速度", "right_wrist_speed")

    for feature_name, label in (
        ("left_wrist_speed", "左手腕"),
        ("right_wrist_speed", "右手腕"),
        ("shoulder_hip_separation_angle_deg", "肩髋分离角"),
    ):
        window_items = windows.get(feature_name) or []
        if not window_items:
            continue
        peak = window_items[0]
        lines.append(
            f"- {label}关键峰值窗口: frame {peak['peak_frame_index']} ({peak['peak_timestamp_sec']}s) 峰值 {peak['peak_value']}"
        )

    deltas = coordination.get("time_deltas_sec", {})
    if deltas.get("hip_to_shoulder_rotation_peak") is not None:
        lines.append(
            f"- 动力链时序: 髋旋转峰到肩旋转峰时间差 {deltas['hip_to_shoulder_rotation_peak']}s，"
            f"肩旋转峰到主导手腕速度峰时间差 {deltas.get('shoulder_rotation_to_wrist_speed_peak')}s"
        )

    if phases:
        lines.append("- 动作阶段分段:")
        for phase in phases:
            lines.append(
                f"  {phase['phase']}: frame {phase['start_frame']} 到 {phase['end_frame']} "
                f"({phase['start_time_sec']}s - {phase['end_time_sec']}s)"
            )

    lines.extend(
        [
            "",
            "建议给大模型的分析方向：",
            "1. 先用肩髋分离角、肩/髋旋转速度和主导手腕速度峰值判断蓄力顺序与发力链是否顺畅。",
            "2. 再结合动作阶段分段，分析准备、加速挥拍、收拍、回位是否节奏清晰。",
            "3. 结合站位宽度、膝角和中心速度判断下肢参与度、重心转移和制动能力。",
            "4. 结合头部稳定性与左右对称性，观察是否存在头部晃动、偏侧代偿或上肢过度主导。",
            "5. 本摘要仅基于人体关键点，球拍/球相关特征仍需额外检测器支持。",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else input_path.with_name(f"{input_path.stem}_summary.txt")
    )

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    payload = load_payload(input_path)
    frames = payload.get("frames", [])
    segments = visibility_segments(frames)
    visible = present_frames(frames)

    stats = {feature_name: calc_stat_triplet(feature_values(frames, feature_name)) for feature_name in KEY_FEATURES}
    peaks = {
        "left_wrist_speed": top_events(frames, "left_wrist_speed"),
        "right_wrist_speed": top_events(frames, "right_wrist_speed"),
        "shoulder_rotation_velocity_deg_per_sec": top_events(frames, "shoulder_rotation_velocity_deg_per_sec", by_abs=True),
        "hip_rotation_velocity_deg_per_sec": top_events(frames, "hip_rotation_velocity_deg_per_sec", by_abs=True),
        "shoulder_hip_separation_angle_deg": top_events(frames, "shoulder_hip_separation_angle_deg"),
        "shoulder_hip_separation_angular_velocity_deg_per_sec": top_events(
            frames,
            "shoulder_hip_separation_angular_velocity_deg_per_sec",
            by_abs=True,
        ),
        "center_speed": top_events(frames, "center_speed"),
        "hip_center_speed": top_events(frames, "hip_center_speed"),
    }
    windows = {
        "left_wrist_speed": peak_window_for_feature(frames, "left_wrist_speed"),
        "right_wrist_speed": peak_window_for_feature(frames, "right_wrist_speed"),
        "shoulder_hip_separation_angle_deg": peak_window_for_feature(frames, "shoulder_hip_separation_angle_deg"),
        "shoulder_rotation_velocity_deg_per_sec": peak_window_for_feature(
            frames,
            "shoulder_rotation_velocity_deg_per_sec",
            by_abs=True,
        ),
        "hip_rotation_velocity_deg_per_sec": peak_window_for_feature(
            frames,
            "hip_rotation_velocity_deg_per_sec",
            by_abs=True,
        ),
    }
    stance_counts = stance_distribution(visible)
    dominant = dominant_side(payload)
    dominant_wrist_feature = f"{dominant['dominant_wrist_speed_side']}_wrist_speed" if dominant.get("dominant_wrist_speed_side") else "right_wrist_speed"
    coordination = coordination_metrics(visible, dominant_wrist_feature)
    phases = action_phase_segments(frames, segments, dominant_wrist_feature)

    output_payload = {
        "source_file": str(input_path),
        "video_summary": payload.get("summary", {}),
        "visibility_segments": segments,
        "stance_type_distribution": stance_counts,
        "dominant_side": dominant,
        "feature_stats": stats,
        "peak_events": peaks,
        "peak_windows": windows,
        "coordination_metrics": coordination,
        "head_stability_summary": head_stability_summary(visible),
        "symmetry_summary": symmetry_summary(visible),
        "action_phase_segments": phases,
        "unavailable_without_extra_detectors": unavailable_without_extra_detectors(),
        "llm_brief": build_llm_brief(
            payload,
            segments,
            stats,
            peaks,
            windows,
            stance_counts,
            dominant,
            coordination,
            phases,
        ),
    }

    output_path.write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"已保存摘要: {output_path}")


if __name__ == "__main__":
    main()
