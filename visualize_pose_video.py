import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image


def transcode_to_h264(src_path: Path, dst_path: Path) -> bool:
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return False
    tmp_path = dst_path.with_suffix(".h264_tmp.mp4")
    try:
        result = subprocess.run(
            [
                ffmpeg_bin,
                "-y",
                "-i",
                str(src_path),
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
        tmp_path.replace(dst_path)
        return True
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        return False


COCO_EDGES = [
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="调用 RTMPose API 并将关键点绘制到视频上")
    parser.add_argument("video", help="输入视频路径")
    parser.add_argument(
        "--output",
        default="pose_result.mp4",
        help="输出视频路径，默认: pose_result.mp4",
    )
    parser.add_argument(
        "--txt-output",
        default="",
        help="关键点序列文本输出路径，默认与输出视频同名 .txt",
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8005/predict",
        help="预测接口地址，默认: http://127.0.0.1:8005/predict",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.15,
        help="关键点可视化阈值，默认: 0.15",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="每隔多少帧做一次推理，默认: 1",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="最多处理多少帧，默认: 0 表示处理完整视频",
    )
    parser.add_argument(
        "--predict-mode",
        choices=("http", "local"),
        default="http",
        help="关键点推理模式：http 通过 /predict 接口调用，local 直接在本地进程推理",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=900.0,
        help="HTTP 推理模式下的单帧请求超时时间（秒），默认: 900",
    )
    return parser.parse_args()


def call_predict_api(frame_bgr: np.ndarray, url: str, timeout_sec: float) -> dict:
    ok, encoded = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        raise RuntimeError("视频帧编码失败")

    response = requests.post(
        url,
        files={"file": ("frame.jpg", encoded.tobytes(), "image/jpeg")},
        timeout=timeout_sec,
    )
    response.raise_for_status()
    return response.json()


def call_predict_local(frame_bgr: np.ndarray) -> dict:
    from app import predict_image

    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return predict_image(Image.fromarray(image_rgb))


def draw_pose(frame_bgr: np.ndarray, result: dict, score_threshold: float) -> np.ndarray:
    people = result.get("people", [])

    for person in people:
        bbox = [int(round(v)) for v in person["bbox_xyxy"]]
        keypoints = person["keypoints"]
        scores = person["scores"]

        cv2.rectangle(
            frame_bgr,
            (bbox[0], bbox[1]),
            (bbox[2], bbox[3]),
            (0, 255, 0),
            2,
        )

        for start_idx, end_idx in COCO_EDGES:
            if scores[start_idx] < score_threshold or scores[end_idx] < score_threshold:
                continue
            start = tuple(int(round(v)) for v in keypoints[start_idx])
            end = tuple(int(round(v)) for v in keypoints[end_idx])
            cv2.line(frame_bgr, start, end, (0, 255, 255), 2)

        for idx, (point, score) in enumerate(zip(keypoints, scores)):
            if score < score_threshold:
                continue
            x_coord, y_coord = (int(round(v)) for v in point)
            cv2.circle(frame_bgr, (x_coord, y_coord), 4, (0, 0, 255), -1)
            cv2.circle(frame_bgr, (x_coord, y_coord), 4, (255, 255, 255), 1)
            cv2.putText(
                frame_bgr,
                str(idx),
                (x_coord + 4, y_coord + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

    cv2.putText(
        frame_bgr,
        f"people: {result.get('num_people', 0)}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return frame_bgr


def build_frame_record(frame_index: int, timestamp_sec: float, result: dict) -> dict:
    people_records = []
    for person_index, person in enumerate(result.get("people", [])):
        people_records.append(
            {
                "person_index": person_index,
                "bbox_xyxy": [round(float(value), 3) for value in person["bbox_xyxy"]],
                "keypoints": [
                    [round(float(coord), 3) for coord in point]
                    for point in person["keypoints"]
                ],
                "scores": [round(float(score), 4) for score in person["scores"]],
            }
        )

    return {
        "frame_index": frame_index,
        "timestamp_sec": round(timestamp_sec, 4),
        "num_people": int(result.get("num_people", len(people_records))),
        "detector_enabled": bool(result.get("detector_enabled", False)),
        "people": people_records,
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.video).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    txt_output_path = (
        Path(args.txt_output).expanduser().resolve()
        if args.txt_output
        else output_path.with_suffix(".txt")
    )

    if not input_path.exists():
        raise FileNotFoundError(f"输入视频不存在: {input_path}")
    if args.frame_step < 1:
        raise ValueError("--frame-step 必须 >= 1")

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    txt_output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_output_path = output_path.with_suffix(".mp4v_tmp.mp4")
    writer = cv2.VideoWriter(
        str(raw_output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"无法创建输出视频: {raw_output_path}")

    last_result: dict | None = None
    frame_index = 0
    processed_frames = 0
    sequence_records: list[dict] = []
    cleanup_runtime = None

    if args.predict_mode == "local":
        from app import ensure_models_loaded, unload_models

        ensure_models_loaded()
        cleanup_runtime = unload_models

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if args.max_frames > 0 and processed_frames >= args.max_frames:
                break

            if frame_index % args.frame_step == 0 or last_result is None:
                if args.predict_mode == "local":
                    last_result = call_predict_local(frame)
                else:
                    last_result = call_predict_api(frame, args.url, args.request_timeout)

            rendered = draw_pose(frame.copy(), last_result, args.score_threshold)
            writer.write(rendered)
            sequence_records.append(
                build_frame_record(
                    frame_index=frame_index,
                    timestamp_sec=frame_index / fps if fps > 0 else 0.0,
                    result=last_result,
                )
            )

            frame_index += 1
            processed_frames += 1

            if processed_frames % 10 == 0:
                total_info = total_frames if total_frames > 0 else "unknown"
                print(f"已处理 {processed_frames}/{total_info} 帧")
    finally:
        capture.release()
        writer.release()
        if cleanup_runtime is not None:
            cleanup_runtime()

    txt_output_path.write_text(
        json.dumps(
            {
                "video_path": str(input_path),
                "output_video_path": str(output_path),
                "fps": round(float(fps), 4),
                "frame_width": width,
                "frame_height": height,
                "total_frames_in_source": total_frames,
                "processed_frames": processed_frames,
                "frame_step": args.frame_step,
                "score_threshold": args.score_threshold,
                "frames": sequence_records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    h264_ok = False
    if raw_output_path.exists() and raw_output_path.stat().st_size > 0:
        h264_ok = transcode_to_h264(raw_output_path, output_path)
    if not h264_ok:
        if raw_output_path.exists():
            raw_output_path.replace(output_path)
        print("[warn] ffmpeg H.264 转码失败，使用 mp4v 原始输出（浏览器可能无法播放）")
    else:
        try:
            if raw_output_path.exists():
                raw_output_path.unlink()
        except Exception:
            pass
        print("[info] 已通过 ffmpeg 转码为 H.264，浏览器可直接播放")

    print(f"已保存视频结果: {output_path}")
    print(f"已保存关键点序列: {txt_output_path}")
    print(f"共处理帧数: {processed_frames}")


if __name__ == "__main__":
    main()
