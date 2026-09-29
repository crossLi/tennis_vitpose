import argparse
import json
import math
from pathlib import Path


KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 RTMPose 关键点序列中计算结构化特征")
    parser.add_argument("input", help="输入关键点序列文本路径")
    parser.add_argument(
        "--output",
        default="",
        help="输出特征文本路径，默认与输入同名并追加 _features.txt",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.15,
        help="关键点有效性阈值，默认: 0.15",
    )
    return parser.parse_args()


def load_sequence(input_path: Path) -> dict:
    return json.loads(input_path.read_text(encoding="utf-8"))


def safe_point(person: dict, index: int, score_threshold: float):
    if not person:
        return None
    scores = person.get("scores", [])
    keypoints = person.get("keypoints", [])
    if index >= len(scores) or index >= len(keypoints):
        return None
    if scores[index] < score_threshold:
        return None
    point = keypoints[index]
    return float(point[0]), float(point[1])


def midpoint(point_a, point_b):
    if point_a is None or point_b is None:
        return None
    return ((point_a[0] + point_b[0]) / 2.0, (point_a[1] + point_b[1]) / 2.0)


def distance(point_a, point_b):
    if point_a is None or point_b is None:
        return None
    return math.dist(point_a, point_b)


def axis_delta(point_a, point_b):
    if point_a is None or point_b is None:
        return None, None
    return point_a[0] - point_b[0], point_a[1] - point_b[1]


def angle_degrees(point_a, vertex, point_c):
    if point_a is None or vertex is None or point_c is None:
        return None
    vec1 = (point_a[0] - vertex[0], point_a[1] - vertex[1])
    vec2 = (point_c[0] - vertex[0], point_c[1] - vertex[1])
    norm1 = math.hypot(*vec1)
    norm2 = math.hypot(*vec2)
    if norm1 == 0 or norm2 == 0:
        return None
    cosine = max(-1.0, min(1.0, (vec1[0] * vec2[0] + vec1[1] * vec2[1]) / (norm1 * norm2)))
    return math.degrees(math.acos(cosine))


def line_angle_degrees(point_a, point_b):
    if point_a is None or point_b is None:
        return None
    return math.degrees(math.atan2(point_b[1] - point_a[1], point_b[0] - point_a[0]))


def separation_angle_degrees(angle_a, angle_b):
    if angle_a is None or angle_b is None:
        return None
    diff = abs(angle_a - angle_b) % 180.0
    if diff > 90.0:
        diff = 180.0 - diff
    return diff


def signed_angle_delta_deg(current_angle, previous_angle):
    if current_angle is None or previous_angle is None:
        return None
    return (current_angle - previous_angle + 180.0) % 360.0 - 180.0


def angle_velocity_deg_per_sec(current_angle, previous_angle, delta_t):
    delta = signed_angle_delta_deg(current_angle, previous_angle)
    if delta is None or delta_t <= 0:
        return None
    return delta / delta_t


def scalar_velocity_per_sec(current_value, previous_value, delta_t):
    if current_value is None or previous_value is None or delta_t <= 0:
        return None
    return (current_value - previous_value) / delta_t


def vector_speed(current_point, previous_point, delta_t):
    if current_point is None or previous_point is None or delta_t <= 0:
        return None
    return distance(current_point, previous_point) / delta_t


def component_velocity(current_point, previous_point, delta_t):
    if current_point is None or previous_point is None or delta_t <= 0:
        return None, None
    return (
        (current_point[0] - previous_point[0]) / delta_t,
        (current_point[1] - previous_point[1]) / delta_t,
    )


def safe_ratio(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def round_or_none(value, digits=4):
    if value is None:
        return None
    return round(float(value), digits)


def round_point(point, digits=3):
    if point is None:
        return None
    return [round_or_none(point[0], digits), round_or_none(point[1], digits)]


def select_person(frame: dict) -> dict | None:
    people = frame.get("people", [])
    if not people:
        return None
    return max(
        people,
        key=lambda person: (person["bbox_xyxy"][2] - person["bbox_xyxy"][0])
        * (person["bbox_xyxy"][3] - person["bbox_xyxy"][1]),
    )


def infer_stance_type_heuristic(stance_width_ratio, feet_line_angle, hip_line_angle):
    if stance_width_ratio is None:
        return None
    line_gap = separation_angle_degrees(feet_line_angle, hip_line_angle)
    if stance_width_ratio < 1.2:
        return "narrow"
    if line_gap is not None and line_gap >= 18:
        return "open"
    if stance_width_ratio >= 1.8:
        return "semi_open"
    return "neutral"


def build_frame_features(frame: dict, previous_features: dict | None, score_threshold: float) -> dict:
    person = select_person(frame)
    timestamp_sec = float(frame.get("timestamp_sec", 0.0))
    result = {
        "frame_index": int(frame.get("frame_index", -1)),
        "timestamp_sec": round(timestamp_sec, 4),
        "person_present": person is not None,
        "num_people": int(frame.get("num_people", 0)),
    }

    if person is None:
        result["features"] = {}
        return result

    points = {name: safe_point(person, idx, score_threshold) for idx, name in enumerate(KEYPOINT_NAMES)}
    mid_shoulder = midpoint(points["left_shoulder"], points["right_shoulder"])
    mid_hip = midpoint(points["left_hip"], points["right_hip"])
    bbox = [float(value) for value in person["bbox_xyxy"]]
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    bbox_center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    prev_features = previous_features["features"] if previous_features and previous_features.get("person_present") else {}
    prev_timestamp = previous_features["timestamp_sec"] if previous_features else None
    delta_t = timestamp_sec - prev_timestamp if prev_timestamp is not None else 0.0

    prev_bbox_center = tuple(prev_features["bbox_center"]) if prev_features.get("bbox_center") else None
    prev_mid_hip = tuple(prev_features["mid_hip"]) if prev_features.get("mid_hip") else None
    prev_left_wrist = tuple(prev_features["left_wrist"]) if prev_features.get("left_wrist") else None
    prev_right_wrist = tuple(prev_features["right_wrist"]) if prev_features.get("right_wrist") else None
    prev_left_ankle = tuple(prev_features["left_ankle"]) if prev_features.get("left_ankle") else None
    prev_right_ankle = tuple(prev_features["right_ankle"]) if prev_features.get("right_ankle") else None
    prev_shoulder_angle = prev_features.get("shoulder_line_angle_deg")
    prev_hip_angle = prev_features.get("hip_line_angle_deg")
    prev_sep_angle = prev_features.get("shoulder_hip_separation_angle_deg")

    shoulder_line_angle = line_angle_degrees(points["left_shoulder"], points["right_shoulder"])
    hip_line_angle = line_angle_degrees(points["left_hip"], points["right_hip"])
    shoulder_hip_sep = separation_angle_degrees(shoulder_line_angle, hip_line_angle)
    feet_line_angle = line_angle_degrees(points["left_ankle"], points["right_ankle"])

    shoulder_width = distance(points["left_shoulder"], points["right_shoulder"])
    hip_width = distance(points["left_hip"], points["right_hip"])
    torso_length = distance(mid_shoulder, mid_hip)
    stance_width = distance(points["left_ankle"], points["right_ankle"])

    center_vx, center_vy = component_velocity(bbox_center, prev_bbox_center, delta_t)
    hip_vx, hip_vy = component_velocity(mid_hip, prev_mid_hip, delta_t)

    nose_dx_mid_hip, nose_dy_mid_hip = axis_delta(points["nose"], mid_hip)
    wrist_dx, wrist_dy = axis_delta(points["left_wrist"], points["right_wrist"])
    ankle_dx, ankle_dy = axis_delta(points["left_ankle"], points["right_ankle"])

    left_arm_length = (
        (distance(points["left_shoulder"], points["left_elbow"]) or 0.0)
        + (distance(points["left_elbow"], points["left_wrist"]) or 0.0)
    )
    right_arm_length = (
        (distance(points["right_shoulder"], points["right_elbow"]) or 0.0)
        + (distance(points["right_elbow"], points["right_wrist"]) or 0.0)
    )
    left_leg_length = (
        (distance(points["left_hip"], points["left_knee"]) or 0.0)
        + (distance(points["left_knee"], points["left_ankle"]) or 0.0)
    )
    right_leg_length = (
        (distance(points["right_hip"], points["right_knee"]) or 0.0)
        + (distance(points["right_knee"], points["right_ankle"]) or 0.0)
    )

    features = {
        "bbox_center": round_point(bbox_center),
        "bbox_width": round_or_none(bbox_width, 3),
        "bbox_height": round_or_none(bbox_height, 3),
        "bbox_area": round_or_none(bbox_width * bbox_height, 3),
        "mid_shoulder": round_point(mid_shoulder),
        "mid_hip": round_point(mid_hip),
        "shoulder_width": round_or_none(shoulder_width, 3),
        "hip_width": round_or_none(hip_width, 3),
        "torso_length": round_or_none(torso_length, 3),
        "torso_angle_deg": round_or_none(line_angle_degrees(mid_shoulder, mid_hip), 3),
        "shoulder_line_angle_deg": round_or_none(shoulder_line_angle, 3),
        "hip_line_angle_deg": round_or_none(hip_line_angle, 3),
        "feet_line_angle_deg": round_or_none(feet_line_angle, 3),
        "shoulder_hip_separation_angle_deg": round_or_none(shoulder_hip_sep, 3),
        "shoulder_rotation_velocity_deg_per_sec": round_or_none(
            angle_velocity_deg_per_sec(shoulder_line_angle, prev_shoulder_angle, delta_t),
            3,
        ),
        "hip_rotation_velocity_deg_per_sec": round_or_none(
            angle_velocity_deg_per_sec(hip_line_angle, prev_hip_angle, delta_t),
            3,
        ),
        "shoulder_hip_separation_angular_velocity_deg_per_sec": round_or_none(
            scalar_velocity_per_sec(shoulder_hip_sep, prev_sep_angle, delta_t),
            3,
        ),
        "left_elbow_angle_deg": round_or_none(
            angle_degrees(points["left_shoulder"], points["left_elbow"], points["left_wrist"]), 3
        ),
        "right_elbow_angle_deg": round_or_none(
            angle_degrees(points["right_shoulder"], points["right_elbow"], points["right_wrist"]), 3
        ),
        "left_shoulder_angle_deg": round_or_none(
            angle_degrees(points["left_elbow"], points["left_shoulder"], points["left_hip"]), 3
        ),
        "right_shoulder_angle_deg": round_or_none(
            angle_degrees(points["right_elbow"], points["right_shoulder"], points["right_hip"]), 3
        ),
        "left_hip_angle_deg": round_or_none(
            angle_degrees(points["left_shoulder"], points["left_hip"], points["left_knee"]), 3
        ),
        "right_hip_angle_deg": round_or_none(
            angle_degrees(points["right_shoulder"], points["right_hip"], points["right_knee"]), 3
        ),
        "left_knee_angle_deg": round_or_none(
            angle_degrees(points["left_hip"], points["left_knee"], points["left_ankle"]), 3
        ),
        "right_knee_angle_deg": round_or_none(
            angle_degrees(points["right_hip"], points["right_knee"], points["right_ankle"]), 3
        ),
        "left_arm_length": round_or_none(left_arm_length, 3),
        "right_arm_length": round_or_none(right_arm_length, 3),
        "left_leg_length": round_or_none(left_leg_length, 3),
        "right_leg_length": round_or_none(right_leg_length, 3),
        "left_wrist_speed": round_or_none(vector_speed(points["left_wrist"], prev_left_wrist, delta_t), 3),
        "right_wrist_speed": round_or_none(vector_speed(points["right_wrist"], prev_right_wrist, delta_t), 3),
        "left_ankle_speed": round_or_none(vector_speed(points["left_ankle"], prev_left_ankle, delta_t), 3),
        "right_ankle_speed": round_or_none(vector_speed(points["right_ankle"], prev_right_ankle, delta_t), 3),
        "center_speed": round_or_none(vector_speed(bbox_center, prev_bbox_center, delta_t), 3),
        "hip_center_speed": round_or_none(vector_speed(mid_hip, prev_mid_hip, delta_t), 3),
        "center_velocity_x_per_sec": round_or_none(center_vx, 3),
        "center_velocity_y_per_sec": round_or_none(center_vy, 3),
        "hip_center_velocity_x_per_sec": round_or_none(hip_vx, 3),
        "hip_center_velocity_y_per_sec": round_or_none(hip_vy, 3),
        "stance_width": round_or_none(stance_width, 3),
        "stance_width_to_hip_width_ratio": round_or_none(safe_ratio(stance_width, hip_width), 3),
        "stance_width_to_torso_length_ratio": round_or_none(safe_ratio(stance_width, torso_length), 3),
        "stance_type_heuristic": infer_stance_type_heuristic(
            safe_ratio(stance_width, hip_width),
            feet_line_angle,
            hip_line_angle,
        ),
        "head_to_mid_shoulder_distance_px": round_or_none(distance(points["nose"], mid_shoulder), 3),
        "head_to_mid_hip_distance_px": round_or_none(distance(points["nose"], mid_hip), 3),
        "nose_offset_x_from_mid_hip_px": round_or_none(nose_dx_mid_hip, 3),
        "nose_offset_y_from_mid_hip_px": round_or_none(nose_dy_mid_hip, 3),
        "left_right_elbow_angle_diff_deg": round_or_none(
            abs(
                (angle_degrees(points["left_shoulder"], points["left_elbow"], points["left_wrist"]) or 0.0)
                - (angle_degrees(points["right_shoulder"], points["right_elbow"], points["right_wrist"]) or 0.0)
            ),
            3,
        ),
        "left_right_knee_angle_diff_deg": round_or_none(
            abs(
                (angle_degrees(points["left_hip"], points["left_knee"], points["left_ankle"]) or 0.0)
                - (angle_degrees(points["right_hip"], points["right_knee"], points["right_ankle"]) or 0.0)
            ),
            3,
        ),
        "left_right_arm_length_diff_px": round_or_none(abs(left_arm_length - right_arm_length), 3),
        "left_right_leg_length_diff_px": round_or_none(abs(left_leg_length - right_leg_length), 3),
        "wrist_height_diff_px": round_or_none(abs(wrist_dy) if wrist_dy is not None else None, 3),
        "ankle_height_diff_px": round_or_none(abs(ankle_dy) if ankle_dy is not None else None, 3),
    }

    for key in ("left_wrist", "right_wrist", "left_ankle", "right_ankle", "nose"):
        features[key] = round_point(points[key])

    result["features"] = features
    return result


def summarize_features(frame_features: list[dict]) -> dict:
    present_frames = [item for item in frame_features if item["person_present"]]
    coverage = len(present_frames) / len(frame_features) if frame_features else 0.0

    def avg(key: str):
        values = [item["features"].get(key) for item in present_frames if item["features"].get(key) is not None]
        if not values:
            return None
        return round(sum(values) / len(values), 3)

    stance_counts = {}
    for item in present_frames:
        stance = item["features"].get("stance_type_heuristic")
        if not stance:
            continue
        stance_counts[stance] = stance_counts.get(stance, 0) + 1

    return {
        "total_frames": len(frame_features),
        "person_visible_frames": len(present_frames),
        "person_visible_ratio": round(coverage, 4),
        "avg_bbox_area": avg("bbox_area"),
        "avg_torso_length": avg("torso_length"),
        "avg_torso_angle_deg": avg("torso_angle_deg"),
        "avg_shoulder_hip_separation_angle_deg": avg("shoulder_hip_separation_angle_deg"),
        "avg_shoulder_rotation_velocity_deg_per_sec": avg("shoulder_rotation_velocity_deg_per_sec"),
        "avg_hip_rotation_velocity_deg_per_sec": avg("hip_rotation_velocity_deg_per_sec"),
        "avg_shoulder_hip_separation_angular_velocity_deg_per_sec": avg(
            "shoulder_hip_separation_angular_velocity_deg_per_sec"
        ),
        "avg_left_elbow_angle_deg": avg("left_elbow_angle_deg"),
        "avg_right_elbow_angle_deg": avg("right_elbow_angle_deg"),
        "avg_left_knee_angle_deg": avg("left_knee_angle_deg"),
        "avg_right_knee_angle_deg": avg("right_knee_angle_deg"),
        "avg_center_speed": avg("center_speed"),
        "avg_hip_center_speed": avg("hip_center_speed"),
        "avg_left_wrist_speed": avg("left_wrist_speed"),
        "avg_right_wrist_speed": avg("right_wrist_speed"),
        "avg_left_ankle_speed": avg("left_ankle_speed"),
        "avg_right_ankle_speed": avg("right_ankle_speed"),
        "avg_stance_width": avg("stance_width"),
        "avg_stance_width_to_hip_width_ratio": avg("stance_width_to_hip_width_ratio"),
        "avg_head_to_mid_hip_distance_px": avg("head_to_mid_hip_distance_px"),
        "avg_left_right_elbow_angle_diff_deg": avg("left_right_elbow_angle_diff_deg"),
        "avg_left_right_knee_angle_diff_deg": avg("left_right_knee_angle_diff_deg"),
        "stance_type_counts": stance_counts,
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else input_path.with_name(f"{input_path.stem}_features.txt")
    )

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    sequence = load_sequence(input_path)
    frame_features = []
    previous = None

    for frame in sequence.get("frames", []):
        current = build_frame_features(frame, previous, args.score_threshold)
        frame_features.append(current)
        previous = current

    payload = {
        "source_file": str(input_path),
        "fps": sequence.get("fps"),
        "frame_width": sequence.get("frame_width"),
        "frame_height": sequence.get("frame_height"),
        "score_threshold": args.score_threshold,
        "summary": summarize_features(frame_features),
        "frames": frame_features,
    }

    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"已保存关键点特征: {output_path}")


if __name__ == "__main__":
    main()
