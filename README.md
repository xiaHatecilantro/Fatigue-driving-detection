# 驾驶员疲劳与分心检测系统

融合**传统视觉规则**和**深度学习模型**的驾驶员状态检测全栈系统。通过 MediaPipe 提取人脸关键点计算 EAR/MAR/头部姿态进行时序打分，同时用 MobileNetV3 分类器进行视觉判断，最终由融合引擎综合两者的输出。

## 技术栈

| 层 | 技术 |
|---|------|
| 前端 | React + TypeScript + Vite |
| 后端 | FastAPI + Pydantic |
| 计算机视觉 | MediaPipe Face Mesh + OpenCV |
| 深度学习 | PyTorch + MobileNetV3-Small + torchvision |
| 配置 | YAML |

## 项目结构

```
├── backend/           # FastAPI 后端服务
│   ├── api/routes/    # HTTP 路由层（health / infer / metrics / ws）
│   ├── services/      # 业务编排层（推理服务 + 训练指标服务）
│   └── schemas/       # Pydantic 数据模型
├── configs/           # YAML 配置文件（阈值、权重、摄像头参数）
├── cv/                # 传统视觉特征提取与规则打分
│   ├── features/      # EAR / MAR / 头部姿态计算
│   └── scoring/       # 规则法打分器（时序逻辑 + 风险累加）
├── inference/         # 统一推理管线与融合引擎
│   ├── common_pipeline.py   # 单帧推理总控
│   ├── fusion_engine.py     # 规则 + 模型融合
│   ├── model_runner.py      # MobileNetV3 推理封装
│   ├── image_infer.py       # 离线图片推理入口
│   ├── video_infer.py       # 离线视频推理入口
│   └── realtime_detector.py # 摄像头实时检测
├── training/          # 分类模型训练与评估
│   ├── configs/       # 训练配置 + 数据集映射规则
│   ├── datasets/      # PyTorch Dataset
│   └── models/        # MobileNetV3 模型定义
├── frontend/          # React 前端
│   └── src/
│       ├── components/  # UploadPanel / ResultCard / TrainingMetricsPanel / RealtimePanel
│       ├── services/    # HTTP 请求封装（fetch API）
│       ├── types/       # TypeScript 类型定义
│       └── hooks/       # WebSocket 自定义 Hook
├── scripts/           # 数据处理脚本
└── tests/             # 测试
```

## 核心设计

### 规则 + 模型双引擎融合

```
摄像头画面
    ├─→ MediaPipe → 468点关键点 → EAR/MAR/头部角度 → 规则打分（70%）──┐
    │                                                                   ├─→ 融合决策
    └─→ MobileNetV3（分类器） → 4类概率 → 模型打分（30%）────────────┘
```

- **规则法**：计算眼睛宽高比（EAR）、嘴巴宽高比（MAR）、头部偏转角（yaw/pitch/roll），通过连续帧计数过滤偶发噪声，累加减分
- **模型法**：MobileNetV3-Small 四分类器（normal / eye_closed / yawn / distracted），输出各类概率
- **融合引擎**：结合两者结果，规则为主（稳定可解释），模型为辅（补充视觉模式），支持图片/视频/实时三种模式自适应权重

### 风险等级

| 等级 | 分数范围 | 说明 |
|------|---------|------|
| normal | 0-30 | 正常驾驶 |
| mild | 30-60 | 轻度风险 |
| moderate | 60-80 | 中度风险 |
| severe | 80-100 | 重度风险 |

## 快速开始

### 环境要求

- Python 3.10 ~ 3.11（推荐 conda 环境）
- Node.js 18+
- 摄像头（实时模式需要）

### 1. 安装后端依赖

```bash
conda create -n driving python=3.11
conda activate driving
pip install fastapi uvicorn mediapipe==0.10.9 opencv-python pyyaml pydantic torch torchvision numpy pillow python-multipart
```

### 2. 安装前端依赖

```bash
cd frontend
npm install
```

### 3. 启动后端

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

访问 `http://127.0.0.1:8000/docs` 查看交互式 API 文档。

### 4. 启动前端

```bash
cd frontend
npm run dev
```

浏览器打开 `http://localhost:5173`。

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 根路径 |
| GET | `/health` | 健康检查 |
| POST | `/api/infer/image` | 上传图片，返回疲劳/分心检测结果 |
| POST | `/api/infer/video` | 上传视频，返回逐帧结果 + 事件列表 + 汇总 |
| GET | `/api/metrics/training` | 获取模型训练指标和混淆矩阵 |
| WS | `/ws/realtime` | WebSocket 实时摄像头检测 |

## 模型训练

### 数据集

训练数据按类别目录组织：

```
data/processed/unified_dataset/
├── train/
│   ├── normal/
│   ├── eye_closed/
│   ├── yawn/
│   └── distracted/
└── val/
    └── ...
```

### 训练命令

```bash
python -m training.train_classifier --config training/configs/base.yaml
```

### 评估命令

```bash
python -m training.eval_classifier --config training/configs/base.yaml --checkpoint path/to/best.pt
```

### 当前模型指标

| 指标 | 值 |
|------|------|
| 准确率 | 97.80% |
| 宏平均 F1 | 97.92% |
| 闭眼 F1 | 99.50% |
| 哈欠 F1 | 99.50% |
| 分心 F1 | 95.38% |

## 配置说明

核心阈值和权重集中在 `configs/mvp.yaml`：

```yaml
thresholds:
  ear_closed: 0.22      # 低于此值判定闭眼
  mar_yawn: 0.60        # 高于此值判定张嘴（哈欠）

temporal:
  eye_closed_frames: 3  # 连续闭眼≥3帧才触发
  yawn_frames: 8        # 连续张嘴≥8帧才触发

fusion:
  weights:
    rule_weight: 0.4    # 规则法权重
    model_weight: 0.6   # 模型法权重
  image_weights:        # 图片模式专用权重（无时序信息）
    rule_weight: 0.2
    model_weight: 0.8
```

## 注意事项

- **MediaPipe 版本**：项目使用 `mediapipe==0.10.9`，更高版本已移除 `solutions` API，会导致不兼容
- **Python 版本**：建议 3.11，3.12+ 可能存在兼容性问题
- **模型权重**：推理需要 `training/outputs/` 下的 checkpoint 文件，若不存在系统会降级为纯规则模式
- **数据集**：源码仓库不包含训练数据和模型权重文件，需自行准备或训练

## 适用场景

- 学习"AI 模型如何嵌入完整应用"的教学演示
- 驾驶员状态监测原型验证
- 计算机视觉 + Web 全栈开发参考
