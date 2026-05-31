# 驾驶员疲劳与分心检测系统

融合**传统视觉规则**（MediaPipe 人脸关键点 + EAR/MAR/头部姿态时序打分）和 **YOLO11m-cls 深度学习模型**的驾驶员状态检测全栈系统。

## 项目概况

| 项目 | 说明 |
|------|------|
| 检测对象 | 驾驶员疲劳（闭眼/哈欠）与分心（低头/转头） |
| 检测方式 | 规则法（EAR/MAR/头部姿态时序打分）+ 模型法（YOLO11m 四分类）融合 |
| 模型 | YOLO11m-cls，10.4M 参数，94.3% 准确率 |
| 数据 | 9751 张人工标注全景驾驶图像（Yi Li et al., IEEE ITSC 2025） |
| 技术栈 | FastAPI + React + TypeScript + MediaPipe + ultralytics + PyTorch |

## 核心设计

### 双引擎融合架构

```
摄像头画面
    ├─→ MediaPipe → 468点人脸关键点 → EAR/MAR/头部角度 → 规则打分 ──┐
    │                                                                  ├─→ 融合决策
    └─→ YOLO11m-cls → 4类概率 → 模型打分 ──────────────────────────┘
```

- **规则法**：计算眼睛宽高比（EAR）、嘴巴宽高比（MAR）、头部偏转角（yaw/pitch/roll），连续帧过滤偶发噪声，累加评分
- **模型法**：YOLO11m 四分类器（normal / eye_closed / yawn / distracted），支持全景图直接推理
- **融合引擎**：图片模式模型优先（80%权重），视频模式规则保留话语权（40%），按场景自适应

### 风险等级

| 等级 | 分数范围 | 说明 |
|------|---------|------|
| normal | 0-30 | 正常驾驶 |
| mild | 30-60 | 轻度风险，提醒注意 |
| moderate | 60-80 | 中度风险，建议休息 |
| severe | 80-100 | 重度风险，告警 |

## 项目结构

```
├── backend/              # FastAPI 后端
│   ├── api/routes/       # HTTP 路由（health / infer / metrics / ws）
│   ├── services/         # 业务编排（推理服务 + 训练指标服务）
│   └── schemas/          # Pydantic 数据模型
├── configs/              # 运行配置（融合权重、阈值、摄像头参数）
├── cv/                   # 传统视觉（EAR/MAR/头部姿态 + 规则打分）
├── inference/            # 推理管线 + 融合引擎 + 模型运行器
├── training/             # 训练入口 + 配置
│   └── configs/          # YOLO11 / MobileNetV3（废弃）训练配置
├── frontend/             # React 前端
│   └── src/
│       ├── components/   # UploadPanel / ResultCard / TrainingMetricsPanel / RealtimePanel
│       ├── services/     # HTTP 请求封装（fetch）
│       └── types/        # TypeScript 类型定义
├── scripts/              # 数据处理脚本
├── tests/                # 测试
└── requirements.txt      # Python 依赖清单
```

## 快速开始

### 环境要求

- Python 3.11（推荐 conda）
- Node.js 18+
- NVIDIA GPU（可选，CPU 也可运行）
- 摄像头（实时模式需要）

### 1. 安装依赖

```bash
conda create -n driving python=3.11
conda activate driving
pip install -r requirements.txt
```

### 2. 确保模型权重存在

```bash
# 训练好的模型应位于：
ls runs/classify/training/outputs/yolo11m_newdata_baseline/weights/best.pt
```

### 3. 启动后端

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

API 文档：`http://127.0.0.1:8000/docs`

### 4. 启动前端

```bash
cd frontend
npm install
npm run dev
```

打开 `http://localhost:5173`

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/api/infer/image` | 图片疲劳/分心检测 |
| POST | `/api/infer/video` | 视频逐帧检测 + 事件列表 |
| GET | `/api/metrics/training` | 模型训练指标和混淆矩阵 |
| WS | `/ws/realtime` | WebSocket 实时摄像头检测 |

## 模型训练

### 当前模型

| 指标 | 值 |
|------|------|
| 架构 | YOLO11m-cls |
| 参数 | 10.4M |
| 训练数据 | 9751 张全景图（Yi Li et al., IEEE ITSC 2025） |
| 类别 | normal / eye_closed / yawn / distracted |
| 准确率 | 94.3% |
| 训练环境 | NVIDIA RTX 4060, 15 epoch, ~40 分钟 |

### 训练命令

```bash
# 四分类全图训练
python training/train_classifier.py --config training/configs/yolo_newdata.yaml
```

### 导出模型

```bash
# 一行命令导出到 ONNX
yolo export model=runs/classify/training/outputs/yolo11m_newdata_baseline/weights/best.pt format=onnx

# 支持的导出格式：
# torchscript, onnx, openvino, tensorrt, coreml, tflite, ncnn
```

### 独立部署

`best.pt` 是自包含文件，只需要 `ultralytics` + `torch`，不依赖本项目任何代码：

```python
from ultralytics import YOLO
model = YOLO("best.pt")
result = model("任意图片.jpg")
print(result[0].probs)  # 四类概率
```

## 配置说明

核心参数在 `configs/mvp.yaml`：

```yaml
# 规则法
thresholds:
  ear_closed: 0.22      # 低于此值判定闭眼
  mar_yawn: 0.60        # 高于此值判定哈欠

# 时序过滤（防误报）
temporal:
  eye_closed_frames: 3  # 连续闭眼 ≥ 3 帧才触发
  yawn_frames: 8        # 连续张嘴 ≥ 8 帧才判定

# 融合权重
fusion:
  weights:              # 视频/实时模式
    rule_weight: 0.4
    model_weight: 0.6
  image_weights:        # 图片模式（无时序，模型优先）
    rule_weight: 0.2
    model_weight: 0.8
```

## 更新日志

### 2026-05 — 模型升级与数据集替换

- 分类模型 MobileNetV3 → YOLO11m-cls
- 数据集从 2000 张关键词归类图 → 9751 张人工标注全景图
- 发现并确认旧数据集 yawn 类存在系统性标注错误
- 支持全景图直接推理，模型可独立部署
- 融合引擎新增图片/视频模式自适应权重
- 完善依赖清单（requirements.txt）

### 2026-04 — 项目初始化

- 规则法（EAR/MAR/Head Pose）+ 模型法（MobileNetV3）双引擎融合
- FastAPI + React 前后端完整链路
- 摄像头实时 WebSocket 检测

## 适用场景

- AI 视觉应用竞赛 / 课设 / 毕设
- 学习"AI 模型全栈落地的完整流程"
- 驾驶员状态监测原型参考
- 计算机视觉 + Web 全栈开发教学

## 参考引用

- 数据集：Yi Li et al., "A Comparative Study of Fatigue Detection Models Based on Temporal Feature Fusion," IEEE ITSC, 2025
- 人脸关键点：MediaPipe Face Mesh（Google）
- 分类模型：YOLO11-cls（Ultralytics）
- EAR 公式：Soukupová & Čech, "Real-Time Eye Blink Detection using Facial Landmarks," CVPR 2016
