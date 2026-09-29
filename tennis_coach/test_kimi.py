import argparse
from pathlib import Path

from tennis_coach.kimi_client import analyze_tennis_video, classify_tennis_video


DEFAULT_VIDEO = "/data/lian/vitpose/cam_speed_20260208_195713.avi"
DEFAULT_POSE_VIDEO = "/data/lian/vitpose/cam_speed_pose.mp4"
DEFAULT_POSE_SEQUENCE = "/data/lian/vitpose/cam_speed_pose.txt"
DEFAULT_FEATURES = "/data/lian/vitpose/cam_speed_pose_features.txt"
DEFAULT_SUMMARY = "/data/lian/vitpose/cam_speed_pose_features_summary.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用多模态大模型分析网球动作视频")
    parser.add_argument("--video", default=DEFAULT_VIDEO, help="原始视频路径")
    parser.add_argument("--pose-video", default=DEFAULT_POSE_VIDEO, help="姿态可视化视频路径")
    parser.add_argument("--pose-seq", default=DEFAULT_POSE_SEQUENCE, help="关键点序列路径")
    parser.add_argument("--features", default=DEFAULT_FEATURES, help="关键点特征路径")
    parser.add_argument("--summary", default=DEFAULT_SUMMARY, help="关键点摘要路径")
    parser.add_argument("--provider", default="moonshot", help="模型提供方，如 moonshot / dashscope")
    parser.add_argument("--model", default="", help="模型名；留空时按 provider 使用默认模型")
    parser.add_argument(
        "--output",
        default="/data/lian/vitpose/tennis_coach/kimi_analysis.txt",
        help="分析结果输出路径",
    )
    parser.add_argument(
        "--classify-only",
        action="store_true",
        help="仅做是否为网球视频的判断，不进入完整分析",
    )
    return parser.parse_args()


def require_file(path_str: str, label: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label}不存在: {path}")
    return path


def main() -> None:
    args = parse_args()
    video_path = require_file(args.video, "原始视频")

    classification = classify_tennis_video(video_path=video_path, model=args.model, provider=args.provider)
    print("视频分类结果:")
    print(classification)

    if args.classify_only:
        return

    if not classification["is_tennis_video"]:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("判断不是网球视频，无法进行分析", encoding="utf-8")
        print("判断不是网球视频，无法进行分析")
        return

    pose_video_path = require_file(args.pose_video, "姿态可视化视频")
    pose_seq_path = require_file(args.pose_seq, "关键点序列")
    features_path = require_file(args.features, "关键点特征")
    summary_path = require_file(args.summary, "关键点摘要")

    analysis_text = analyze_tennis_video(
        video_path=video_path,
        pose_video_path=pose_video_path,
        pose_seq_path=pose_seq_path,
        features_path=features_path,
        summary_path=summary_path,
        model=args.model,
        provider=args.provider,
    )

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(analysis_text, encoding="utf-8")

    print(f"已保存分析结果: {output_path}")
    print(analysis_text)


if __name__ == "__main__":
    main()
