# RTMPose 直接部署

## 1. 创建环境

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 2. 放置模型

默认离线读取当前目录下的 `rtmpose-l_8xb256-420e_coco-256x192.py` 和
`rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192-1352a4d2_20230127.pth`
作为 RTMPose 配置和权重。

如果你还下载了人体检测器，请放到：

```bash
models/detector/
```

这样服务会自动启用“先检测人，再做关键点”。
当前会优先读取 `models/detector/yolo26m.pt`，如果没有，再回退到 `models/detector/yolov8n.pt`。

如果只有 `RTMPose` 本体，没有检测器，服务也能启动，但会把整张图当成单人输入。

## 3. 启动服务

默认走 CPU 启动：

```bash
MODEL_DEVICE=cpu uvicorn app:app --host 0.0.0.0 --port 8000
```

如果显卡空闲，可以改成自动选择：

```bash
MODEL_DEVICE=auto uvicorn app:app --host 0.0.0.0 --port 8000
```

如果你想手动指定 RTMPose 配置和权重：

```bash
MODEL_DEVICE=cpu \
RTMPOSE_CONFIG=/home/lian/Documents/trae_projects/vitpose/rtmpose-l_8xb256-420e_coco-256x192.py \
RTMPOSE_CHECKPOINT=/home/lian/Documents/trae_projects/vitpose/rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192-1352a4d2_20230127.pth \
PERSON_DETECTOR_ID=/home/lian/Documents/trae_projects/vitpose/models/detector \
uvicorn app:app --host 0.0.0.0 --port 8000
```

## 4. 健康检查

```bash
curl http://127.0.0.1:8000/health
```

## 5. 推理

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@test.jpg"
```

返回字段：

- `num_people`: 检测到的人数
- `people[].bbox_xyxy`: 人体框
- `people[].keypoints`: 关键点坐标
- `people[].scores`: 每个关键点置信度
- `pose_backend`: 当前姿态模型后端
- `detector_enabled`: 是否启用了本地人体检测器

## 6. 可视化结果图

先启动服务，再运行：

```bash
python visualize_pose.py test.jpg --output pose_result.png
```

如果服务不是本机 `8000` 端口，可以指定：

```bash
python visualize_pose.py test.jpg \
  --url http://127.0.0.1:8001/predict \
  --output pose_result.png
```

默认会在图上绘制：

- 人体框
- 关键点
- COCO 骨架连线
- 关键点编号

## 7. 环境变量

- `MODEL_DEVICE`: `cpu` / `cuda` / `auto`
- `RTMPOSE_CONFIG`: 默认为当前目录下的 `rtmpose-l_8xb256-420e_coco-256x192.py`
- `RTMPOSE_CHECKPOINT`: 默认为当前目录下的 `rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192-1352a4d2_20230127.pth`
- `PERSON_DETECTOR_ID`: 默认为当前目录下优先可用的 `models/detector/yolo26m.pt`
- `DETECTION_THRESHOLD`: 人体检测阈值，默认 `0.3`
- `YOLO_DETECTION_THRESHOLD`: YOLO 人体检测阈值，默认 `0.1`

## 8. RTMPose 依赖

当前服务已经切到 `RTMPose`，除了基础依赖外，还需要安装：

- `mmpose`
- `mmengine`
- `mmcv-lite`
- `mmdet`

如果环境里还没装这些包，服务启动时会直接提示缺少 RTMPose 运行依赖。


# 1. 激活 RTMPose 虚拟环境（必须用这个，包含姿态模型依赖）
source .venv_rtmpose/bin/activate

# 2. 确认前端已构建（我们刚刚已经构建过了 webapp/dist）
ls webapp/dist/index.html   # 应该能看到文件

# 3. 启动服务（CPU模式，推荐先用这个）
MODEL_DEVICE=cpu uvicorn app:app --host 0.0.0.0 --port 8000

# 如果有显卡可用，改用 auto 自动选择：
# MODEL_DEVICE=auto uvicorn app:app --host 0.0.0.0 --port 8000


export TENNIS_LLM_PROVIDER=dashscope
export DASHSCOPE_API_KEY="sk-f0200c7e6be84743ad0303199e80ceb5"
export TENNIS_LLM_PROVIDER=dashscope
MODEL_DEVICE=cpu uvicorn app:app --host 0.0.0.0 --port 8001