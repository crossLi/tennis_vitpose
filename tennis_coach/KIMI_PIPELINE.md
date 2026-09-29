# Kimi 网球动作分析 Pipeline

## 1. 输入清单

推荐给 Kimi 的输入分为 4 层：

1. 原始视频
   - `/data/lian/vitpose/cam_speed_20260208_195713.avi`

2. 姿态可视化视频
   - `/data/lian/vitpose/cam_speed_pose.mp4`
   - 用于人工核对关键点有没有明显漂移

3. 结构化姿态数据
   - `/data/lian/vitpose/cam_speed_pose.txt`
   - 逐帧关键点、bbox、score、timestamp

4. 特征与摘要
   - `/data/lian/vitpose/cam_speed_pose_features.txt`
   - `/data/lian/vitpose/cam_speed_pose_features_summary.txt`

其中真正推荐喂给大模型的核心组合是：

- 原始视频
- 姿态摘要 `cam_speed_pose_features_summary.txt`
- 特征文件节选 `cam_speed_pose_features.txt`
- 关键点序列节选 `cam_speed_pose.txt`

## 2. 完整 Pipeline

### Step 1: 启动 RTMPose 服务

```bash
/data/lian/vitpose/.venv_rtmpose/bin/python3 -m uvicorn app:app --host 0.0.0.0 --port 8000
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

### Step 2: 生成姿态可视化视频与关键点序列

```bash
/data/lian/vitpose/.venv_rtmpose/bin/python3 /data/lian/vitpose/visualize_pose_video.py \
  /data/lian/vitpose/cam_speed_20260208_195713.avi \
  --output /data/lian/vitpose/cam_speed_pose.mp4 \
  --txt-output /data/lian/vitpose/cam_speed_pose.txt \
  --url http://127.0.0.1:8000/predict
```

输出：

- `cam_speed_pose.mp4`
- `cam_speed_pose.txt`

### Step 3: 从关键点序列提取动作特征

```bash
/data/lian/vitpose/.venv_rtmpose/bin/python3 /data/lian/vitpose/extract_pose_features.py \
  /data/lian/vitpose/cam_speed_pose.txt \
  --output /data/lian/vitpose/cam_speed_pose_features.txt
```

输出：

- `cam_speed_pose_features.txt`

### Step 4: 生成适合大模型的摘要

```bash
/data/lian/vitpose/.venv_rtmpose/bin/python3 /data/lian/vitpose/summarize_pose_features.py \
  /data/lian/vitpose/cam_speed_pose_features.txt \
  --output /data/lian/vitpose/cam_speed_pose_features_summary.txt
```

输出：

- `cam_speed_pose_features_summary.txt`

### Step 5: 调用 Kimi 做动作分析

先设置 API Key：

```bash
export MOONSHOT_API_KEY="你的 Kimi API Key"
```

再运行：

```bash
/data/lian/vitpose/.venv_rtmpose/bin/python3 /data/lian/vitpose/tennis_coach/test_kimi.py \
  --video /data/lian/vitpose/cam_speed_20260208_195713.avi \
  --pose-video /data/lian/vitpose/cam_speed_pose.mp4 \
  --pose-seq /data/lian/vitpose/cam_speed_pose.txt \
  --features /data/lian/vitpose/cam_speed_pose_features.txt \
  --summary /data/lian/vitpose/cam_speed_pose_features_summary.txt \
  --output /data/lian/vitpose/tennis_coach/kimi_analysis.txt
```

输出：

- `/data/lian/vitpose/tennis_coach/kimi_analysis.txt`

## 3. 当前脚本实际做了什么

`tennis_coach/test_kimi.py` 现在会：

1. 读取原始视频并转成 base64 `video_url`
2. 读取摘要文件全文
3. 读取特征文件前 6000 字符
4. 读取关键点序列前 4000 字符
5. 拼成一段适合 Kimi 的网球动作分析提示词
6. 调用 `kimi-k2.6`
7. 把结果保存到输出文件

## 4. 当前已纳入的网球分析特征

已纳入 Kimi 提示词的数据重点包括：

- 肩髋分离角
- 肩部旋转速度
- 髋部旋转速度
- 肩髋分离角速度
- 站位宽度与站位类型
- 左右肘/膝角
- 人体中心速度、髋中心速度
- 左右手腕速度
- 头部稳定性
- 左右对称性
- 峰值窗口
- 动作阶段分段
- 动力链关键时间差

## 5. 暂时无法直接计算的特征

以下指标需要额外引入球拍或球检测：

- racket_head_speed
- racket_trajectory
- ball_contact_height
- ball_contact_position_relative_to_body
- racket_body_coordination
- incoming_ball_height_and_timing

## 6. 推荐使用方式

### 第一轮

目标：快速得到动作诊断

输入：

- 原始视频
- 姿态摘要

### 第二轮

目标：深入分析动作链和检测可靠性

输入：

- 原始视频
- 姿态可视化视频
- 摘要
- 特征文件节选

### 第三轮

目标：做精细教练建议

输入：

- Kimi 第一轮分析结果
- 原始视频
- 姿态可视化视频
- 关键峰值窗口
- 动作阶段分段

## 7. 注意事项

1. 当前视频是直接 base64 上传，视频越大，请求越慢。
2. 如果视频很长，建议先裁剪动作片段再发给 Kimi。
3. 当前脚本默认只把特征文件和关键点序列的节选送进模型，避免上下文过长。
4. 如果要提升结论质量，优先确保：
   - 原始视频清晰
   - 姿态可视化没有明显漂点
   - 摘要文件是最新生成的
