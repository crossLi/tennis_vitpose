import argparse
from pathlib import Path

import requests
from PIL import Image, ImageDraw


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
    parser = argparse.ArgumentParser(description="调用 RTMPose API 并将关键点绘制到图片上")
    parser.add_argument("image", help="输入图片路径")
    parser.add_argument(
        "--output",
        default="pose_result.png",
        help="输出图片路径，默认: pose_result.png",
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
    return parser.parse_args()


def call_predict_api(image_path: Path, url: str) -> dict:
    with image_path.open("rb") as fp:
        response = requests.post(
            url,
            files={"file": (image_path.name, fp, "image/png")},
            timeout=300,
        )
    response.raise_for_status()
    return response.json()


def draw_pose(image: Image.Image, result: dict, score_threshold: float) -> Image.Image:
    draw = ImageDraw.Draw(image)
    people = result.get("people", [])

    for person in people:
        bbox = person["bbox_xyxy"]
        keypoints = person["keypoints"]
        scores = person["scores"]

        draw.rectangle(bbox, outline=(0, 255, 0), width=2)

        for start_idx, end_idx in COCO_EDGES:
            if scores[start_idx] < score_threshold or scores[end_idx] < score_threshold:
                continue
            start = tuple(keypoints[start_idx])
            end = tuple(keypoints[end_idx])
            draw.line([start, end], fill=(0, 255, 255), width=3)

        for idx, (point, score) in enumerate(zip(keypoints, scores)):
            if score < score_threshold:
                continue
            x, y = point
            radius = 4
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=(255, 0, 0),
                outline=(255, 255, 255),
                width=1,
            )
            draw.text((x + 4, y + 4), str(idx), fill=(255, 255, 0))

    return image


def main() -> None:
    args = parse_args()
    image_path = Path(args.image).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"输入图片不存在: {image_path}")

    result = call_predict_api(image_path, args.url)
    image = Image.open(image_path).convert("RGB")
    rendered = draw_pose(image, result, args.score_threshold)
    rendered.save(output_path)

    print(f"已保存可视化结果: {output_path}")
    print(f"检测人数: {result.get('num_people', 0)}")
    print(f"detector_enabled: {result.get('detector_enabled')}")


if __name__ == "__main__":
    main()
