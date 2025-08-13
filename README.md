# 模型部署与执行攻击流水线

本项目提供了一个完整的工具集，用于对部署在各种推理引擎（MNN, TFLite, ONNX Runtime）上的深度学习模型进行灰盒对抗性攻击。该流水线涵盖了从环境设置、模型编译、目标筛选到执行复杂自动化攻击的整个流程。

## 项目特点

- **跨引擎支持**: 支持基于 MNN, TFLite, ONNX Runtime 等多种主流推理引擎编译的可执行文件。
- **自动化攻击**: 强大的自动化脚本，能够根据模型可执行文件动态匹配所需的模型资源、Hook 配置和攻击列表。
- **先进的攻击算法**: 采用基于自然进化策略 (NES) 的灰盒攻击方法，无需模型梯度，通过黑盒查询方式高效生成对抗样本。
- **动态聚焦策略**: 实现了先进的“动态聚焦”攻击策略，能够自动、智能地分配攻击资源，优先解决关键的、容易突破的决策边界，显著提高攻击效率和成功率。
- **模块化与可扩展**: 项目结构清晰，易于扩展以支持新的模型、攻击算法或推理引擎。
- **详细的攻击配置**: 提供了丰富的攻击参数，允许用户对攻击过程（如学习率、扰动范围、优化器参数等）进行精细化控制。

## 文件结构概览

```
.
├── CMakeLists.txt              # C++ 项目构建文件 (示例)
├── hook_config/                # GDB Hook 配置文件目录
├── outputs/                    # 攻击结果输出目录
├── pre_attack_scripts/         # 攻击前准备脚本
│   └── generate_false_image_list.sh
├── README.md                   # 项目说明文件
├── resources/                  # 存放模型、可执行文件和图像等资源
│   ├── execution_files/        # 预编译的可执行文件
│   ├── false_image_list/       # “假”图像列表
│   ├── images/                 # 测试图像
│   └── models/                 # 模型文件 (.mnn, .tflite, .onnx 等)
├── scripts/                    # 主要工作流脚本
│   ├── install_dependencies.sh # 依赖安装脚本
│   └── run_automated_attack.sh # 自动化攻击脚本
└── src/                        # 源代码
    └── attackers/              # 攻击算法实现
        ├── nes_attack_targetless.py # 核心 NES 攻击脚本
        └── gdb_script_host.py  # GDB Python 宿主脚本
```

## 快速开始

### 1. 环境准备

本项目建议在 **Ubuntu 24.04** 环境下运行。

首先，运行依赖安装脚本，它将自动安装所有必需的系统依赖和 Python 包。

```bash
bash scripts/install_dependencies.sh
```

该脚本会：
- 安装 `cmake`, `gdb`, `python3`, `opencv` 等系统工具。
- 创建一个名为 `.venv` 的 Python 虚拟环境。
- 在虚拟环境中安装 `numpy`, `opencv-python` 等库。

安装完成后，激活 Python 虚拟环境：

```bash
source .venv/bin/activate
```

### 2. 构建 C++ 可执行文件 (可选)

项目中 `resources/execution_files/` 目录下提供了一些预编译好的可执行文件。如果您需要针对新的模型或自定义逻辑进行编译，可以参考根目录下的 `CMakeLists.txt` 作为模板。

编译一个新的模型通常需要：
1. 编写一个 C++ 主程序（如 `mnist_mnn_console.cpp`），用于加载模型并执行推理。
2. 在 `CMakeLists.txt` 中添加新的 `add_executable` 规则。
3. 确保链接了正确的推理引擎库（如 MNN, TFLite 等）和 OpenCV。

例如，编译一个名为 `my_model_mnn` 的可执行文件：

```cmake
# ... (其他配置)
# 添加 MNN 相关配置
include(cmake/mnn.cmake)

# 添加新的可执行文件
add_executable(my_model_mnn my_model_mnn.cpp)

# 链接库
target_link_libraries(my_model_mnn PRIVATE MNN::MNN ${OpenCV_LIBS})
```

编译完成后，将生成的可执行文件放置在 `resources/execution_files/` 目录中，以便自动化脚本能够找到它。

### 3. 准备攻击目标列表 (关键步骤)

在发起攻击之前，我们需要筛选出一批模型会“误判”的图像。这些图像将作为攻击的起点。

`pre_attack_scripts/generate_false_image_list.sh` 脚本用于自动完成此任务。

**使用方法:**

1.  **准备数据集**: 将您用于筛选的图像数据集放置在一个目录下。
2.  **修改脚本**: 打开 `pre_attack_scripts/generate_false_image_list.sh` 文件，将 `IMAGE_DIR` 变量修改为您的数据集路径。

    ```bash
    # file: pre_attack_scripts/generate_false_image_list.sh
    ...
    # IMPORTANT: Update this path to your image dataset.
    IMAGE_DIR="/path/to/your/image/dataset" # <--- 修改这里
    ...
    ```

3.  **运行脚本**: 执行脚本，并指定一个目标可执行文件。脚本会遍历数据集，将所有被该模型判定为 "false" 的图像路径保存到一个列表中。

    ```bash
    # 示例：为 emotion_ferplus_mnn 生成攻击列表
    bash pre_attack_scripts/generate_false_image_list.sh resources/execution_files/emotion_ferplus_mnn
    ```

脚本执行完毕后，会在 `resources/false_image_list/` 目录下生成一个名为 `<executable_name>_false_list.txt` 的文件。

### 4. 运行自动化攻击

一切准备就绪后，即可启动自动化攻击。

`scripts/run_automated_attack.sh` 脚本是整个攻击流程的核心。它会自动处理所有配置，循环遍历目标列表中的每张图片，并执行 `nes_attack_targetless.py` 脚本进行攻击。

**使用方法:**

只需指定一个目标可执行文件即可。

```bash
# 示例：对 emotion_ferplus_mnn 发起攻击
bash scripts/run_automated_attack.sh resources/execution_files/emotion_ferplus_mnn
```

脚本会自动：
1.  根据可执行文件名，在 `resources/models/` 中找到对应的模型文件。
2.  在 `hook_config/` 中找到对应的 `<executable_name>_hook_config.json` 配置文件。
3.  在 `resources/false_image_list/` 中找到对应的 `<executable_name>_false_list.txt` 列表文件。
4.  遍历列表中的每张图片，调用 Python 攻击脚本。
5.  将每次攻击的结果（包括日志、中间图片、最终对抗样本）保存在 `outputs/<executable_name>_attack_results/` 目录下。

攻击过程中，您会在终端看到详细的日志输出，包括当前的迭代次数、损失值、以及动态聚焦策略的状态。

## 高级配置与自定义

### 自定义攻击参数

`scripts/run_automated_attack.sh` 脚本为 `nes_attack_targetless.py` 设置了默认的攻击参数。您可以直接修改此 shell 脚本来调整参数，以适应不同的模型或攻击目标。

主要的参数包括：
- `--iterations`: 最大攻击迭代次数。
- `--learning-rate`: 初始学习率。
- `--l-inf-norm`: 扰动的 L-无穷范数限制，即允许对原始图像每个像素点做出的最大改动量。
- `--population-size`: NES 算法的种群大小，影响梯度估计的精度和计算开销。
- `--workers`: 用于并行评估的进程数。

**动态聚焦策略参数**:
- `--enable-dynamic-focus`: 启用动态聚焦策略。
- `--boost-weight`: 对当前聚焦的 Hook 点施加的权重，以加速优化。
- `--evaluation-window`: “侦察”模式下的评估窗口大小，用于决定下一个攻击焦点。
- `--satisfaction-patience`: 一个目标点需要连续多少次迭代保持“满足”状态才会被“退休”。

### GDB Hook 配置

本项目的灰盒特性依赖于 GDB 在程序执行时从特定内存地址提取中间状态（如特定层激活值、比较操作的两个值等）。

配置文件位于 `hook_config/` 目录下，为 JSON 格式。每个配置文件对应一个可执行文件。

一个典型的配置项如下：
```json
[
  {
    "address": "0x5555555a7b14",
    "original_branch_instruction": "b.gt",
    "value_pairs": [1],
    "attack_mode": "invert",
    "weight": 1.0
  }
]
```
- `address`: 需要下断点的内存地址。
- `original_branch_instruction`: 该地址原始的 ARMv8 分支指令，如 `b.gt` (大于则跳转)。攻击脚本会根据此指令和 `attack_mode` 来构建损失函数。
- `attack_mode`:
  - `satisfy`: 目标是满足原始的分支条件。
  - `invert`: 目标是反转原始的分支条件（例如，将 `v1 > v2` 的条件反转为 `v1 <= v2`）。
- `weight`: (可选) 该 Hook 点在总损失中的静态权重。

## 贡献

欢迎对本项目进行贡献。如果您有新的攻击想法、支持新的模型或改进现有脚本，请随时提交 Pull Request。





