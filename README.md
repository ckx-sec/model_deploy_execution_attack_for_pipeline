# 模型部署与执行攻击流水线

本项目是一个针对已部署模型的灰盒对抗性攻击工具集，支持 MNN, TFLite, ONNX Runtime 等多种推理引擎。它通过自动化脚本和先进的攻击算法，实现了从环境设置、目标筛选到执行复杂攻击的完整流程。其核心是基于自然进化策略（NES）的攻击算法，并实现了一套创新的“动态聚焦”策略，能够智能分配攻击资源，高效生成对抗样本。

## 文件结构概览

```
.
├── CMakeLists.txt              # C++ 项目构建文件 (示例)
├── hook_config/                # GDB Hook 配置文件目录
├── outputs/                    # 攻击结果输出目录
├── pre_attack_scripts/         # 攻击前准备脚本
├── README.md                   # 项目说明文件
├── resources/                  # 存放模型、可执行文件和图像等资源
├── scripts/                    # 主要工作流脚本
└── src/                        # 源代码
    └── attackers/              # 攻击算法实现
```

## 快速开始

### 1. 环境准备

本项目建议在 **Ubuntu 24.04** 环境下运行。首先，安装所有依赖：

```bash
bash scripts/install_dependencies.sh
```

然后，激活 Python 虚拟环境：

```bash
source .venv/bin/activate
```

### 2. (可选) 构建 C++ 可执行文件

`resources/execution_files/` 目录下已提供预编译文件。如需自行编译，可参考 `CMakeLists.txt` 添加新的编译目标，并将生成的可执行文件放入上述目录。

### 3. 准备攻击目标列表 (关键步骤)

攻击前，需要筛选出模型会“误判”的图像作为攻击起点。

1.  **修改配置**: 打开 `pre_attack_scripts/generate_false_image_list.sh`，将 `IMAGE_DIR` 变量指向您的图像数据集路径。
2.  **运行脚本**: 执行脚本以生成“假”图像列表。

    ```bash
    # 示例: 为 emotion_ferplus_mnn 生成列表
    bash pre_attack_scripts/generate_false_image_list.sh resources/execution_files/emotion_ferplus_mnn
    ```
    生成的列表将位于 `resources/false_image_list/`。

### 4. 运行自动化攻击

准备就绪后，运行主攻击脚本。它会自动匹配模型、Hook配置和目标列表。

```bash
# 示例: 对 emotion_ferplus_mnn 发起攻击
bash scripts/run_automated_attack.sh resources/execution_files/emotion_ferplus_mnn
```

攻击日志和结果将保存在 `outputs/` 目录中。

## 核心攻击策略：动态聚焦 NES

`nes_attack_targetless.py` 脚本采用了一套比标准 NES 更为复杂的、事件驱动的优化策略，旨在高效地解决具有多个决策分支的复杂模型的攻击问题。其核心流程如下：

### 1. 梯度估算与优化器

- **梯度估算 (NES)**: 采用自然进化策略（Natural Evolution Strategies）在不访问模型梯度的情况下，通过对输入图像添加随机扰动并观察损失变化，来估算出一个有效的“梯度”方向。
- **优化器 (Adam)**: 使用 Adam 优化器根据估算出的梯度来更新对抗样本，它结合了一阶矩（动量）和二阶矩（自适应学习率）的优点，使更新过程更稳定高效。

### 2. 动态聚焦策略流程

该策略的核心思想是“集中优势兵力，各个击破”。它并非同时攻击所有的决策分支（Hook点），而是通过一个动态循环的机制来智能地选择当前最重要的目标。

- **阶段一：侦察 (Scouting Mode)**
  - **目标**: 识别出当前最“脆弱”或最容易取得进展的决策分支。
  - **方法**: 在初始阶段，所有未满足的 Hook 点都被赋予一个较低的基础权重。算法会运行一个固定的“评估窗口”（例如10次迭代），并记录每个 Hook 点的损失变化历史。通过线性回归分析损失历史的下降速率，算法可以判断出哪些 Hook 点对当前的扰动最为敏感。

- **阶段二：聚焦 (Focused Fire Mode)**
  - **目标**: 集中火力攻击在“侦察”阶段被识别出的高进展目标。
  - **方法**: 一旦确定了目标，算法会进入“聚焦”模式。它会大幅提升这些目标 Hook 点在总损失函数中的权重（`--boost-weight`），而将其他非目标和已满足的 Hook 点的权重调低。这使得梯度更新主要服务于攻克当前的核心目标。

- **阶段三：目标达成与轮换 (Target Retirement & Rotation)**
  - **目标**: 在一个目标被攻克后，及时将其“退休”，并将资源转移到新的目标上。
  - **方法**: 在“聚焦”模式下，算法会持续监控目标 Hook 点是否已连续多次迭代满足了条件（即损失低于某个阈值）。一旦满足“退休”条件（`--satisfaction-patience`），该目标就会被移出核心攻击列表，其权重也会被调整为较低的“维持”状态。当所有当前焦点目标都被“退休”后，整个系统会自动切回 **阶段一：侦察模式**，去寻找下一批最值得攻击的目标。

这个“侦察-聚焦-退休”的循环过程会一直持续，直到整个攻击任务成功，或者达到最大迭代次数。

### 3. 自适应学习率

为配合动态策略，学习率也采用自适应调整机制。学习率衰减的触发条件有两个：
1.  **固定步数衰减**: 每当迭代达到一定步数（`--lr-decay-steps`），学习率就会衰减。
2.  **停滞衰减**: 如果连续多次迭代（`--stagnation-patience`）中，整体损失没有明显下降，则强制触发一次学习率衰减。

这套完整的策略使得攻击过程更加智能和高效，能够有效应对复杂的真实世界模型。

## 高级配置与自定义

### 自定义攻击参数

您可以通过修改 `scripts/run_automated_attack.sh` 脚本来调整 `nes_attack_targetless.py` 的攻击参数，以适应不同的模型。主要参数包括：
- `--iterations`, `--learning-rate`, `--l-inf-norm`, `--population-size`
- 动态聚焦策略参数: `--enable-dynamic-focus`, `--boost-weight`, `--evaluation-window`, `--satisfaction-patience`

### GDB Hook 配置

灰盒攻击依赖于 `hook_config/` 目录下的 JSON 文件来定义 GDB 断点和损失函数。一个典型的配置项如下：
```json
[
  {
    "address": "0x5555555a7b14",
    "original_branch_instruction": "b.gt",
    "attack_mode": "invert"
  }
]
```
- `address`: 需要下断点的内存地址。
- `original_branch_instruction`: 该地址原始的分支指令，用于构建损失函数。
- `attack_mode`: `satisfy` (满足条件) 或 `invert` (反转条件)。





