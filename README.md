# Shopping Agent Post-Training

> 面向长程工具调用 Agent 的可审计后训练工具箱：**Action-only SFT → 在线 GRPO → 冻结评测**。

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![veRL](https://img.shields.io/badge/veRL-0.8.0-0E8A16)](https://github.com/volcengine/verl)
[![vLLM](https://img.shields.io/badge/vLLM-0.17.0-5A45FF)](https://github.com/vllm-project/vllm)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

购物类 Agent 的难点不在于“输出一个推荐”，而在于在多轮网页/工具交互里稳定完成：检索候选、核验证据、选择正确规格、遵守预算，并在无解时合理终止。本项目沉淀了这一类 Agent 的训练、约束、评测与 GPU 实验治理组件。

它不是一个在线购物产品，也不包含模型权重或商品数据；而是一套可接入 ShopSimulator 风格环境的**后训练与评测工程实现**。

## 项目亮点

- **可执行行为的 SFT**：仅对 Assistant 的工具调用动作计算损失，避免模型学习复述环境 Observation。
- **在线 GRPO 训练适配**：基于 `verl==0.8.0` 接入多轮 Agent Loop、工具定义、Reward 回传与 vLLM rollout。
- **确定性过程验证器**：只读取 Actor 可见的 Observation 与公开工具调用，检查非法动作、重复无进展、候选打开、规格推进和购买就绪；不读取隐藏目标或 Reward 内部细节。
- **受控实验治理**：提供 runtime preflight、失败重放、Canary gate 与无人值守晋级状态机，避免“退出码为 0 即算成功”的不可靠实验结论。
- **可审计评测口径**：结果按固定 task_id 对齐，基础模型、SFT 和 GRPO 使用同一冻结任务集比较；基础设施失败仍保留在分母中。

## 方法概览

```mermaid
flowchart LR
    A[Teacher rollouts] --> B[Reward replay & filtering]
    B --> C[Action-only SFT]
    C --> D[LoRA SFT adapter]
    D --> E[Online GRPO with veRL]
    F[Tool environment] --> E
    E --> G[Frozen evaluation]
    H[Public-state verifier] --> E
    H --> G
    G --> I[Paired metric comparison]
```

训练目标与评测目标分层处理：

| 层级 | 作用 | 边界 |
| --- | --- | --- |
| 终局 Reward | 判断购买/终止结果与约束满足 | 环境定义，训练代码不改写 |
| Process Verifier | 记录公开状态下的动作质量与过程信号 | 不访问隐藏目标，不替代终局 Reward |
| Trajectory Judge | 离线评测轨迹的策略与证据质量 | 不参与在线训练 |

## 实验结果

下表是同一冻结 200 题上的一次确定性 rollout 对比。它用于审计已完成实验，**不表示采样方差、统计显著性或线上效果**。

| 阶段 | 完成率 | 严格成功率 | 购买成功率 | 平均终局 Reward | Guard 拒绝数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-2B Baseline | 18.0% | 0.0% | 0.0% | -0.1105 | 752 |
| LoRA SFT | 96.5% | 60.5% | 60.5% | 0.4729 | 52 |
| GRPO step 100 | 96.5% | **62.0%** | **62.5%** | **0.5158** | **38** |

在该固定口径下，GRPO 比 SFT 多 3 个严格成功任务，并减少了错误购买、循环与 Guard 拒绝。完整聚合结果见 [`results/`](results/)，评测口径见 [`docs/evaluation.md`](docs/evaluation.md)。

## 核心模块

```text
src/shopping_grpo/
├── collection/             # 教师轨迹采集与 Action-only SFT 数据构建
├── environment/            # 工具、Observation、动作与上下文适配
├── evaluation/             # 轨迹规范化、配对比较、评测契约与工件校验
└── training/
    ├── sft/                # LoRA SFT 数据集处理
    └── grpo/               # veRL adapter、GRPO / GraphGPO / GiGPO-lite
        ├── adapter/        # Agent loop、session 与 tools bridge
        ├── process_verifier.py
        ├── dynamic_sampling.py
        ├── graphgpo.py
        └── gigpo.py

scripts/                    # 采集、训练、环境检查、评测与晋级控制器
configs/                    # Agent loop、工具、GRPO 与实验契约
patches/                    # 有版本与 SHA 约束的窄范围 veRL 兼容补丁
tests/                      # 无专有环境即可执行的公开单元测试
results/                    # 聚合指标；不含原始轨迹和任务数据
docs/                       # SFT、GRPO、Reward 与评测设计说明
```

## 快速开始：CPU 开发与测试

公开仓库提供了不依赖 GPU、模型权重或 ShopSimulator 数据的单元测试。建议使用 Python 3.10+：

```bash
git clone https://github.com/go99further/shopping-agent-posttraining.git
cd shopping-agent-posttraining

python -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e . pytest

PYTHONPATH=.:src .venv/bin/python -m pytest -q \
  tests/test_action_validation.py \
  tests/test_dataloader_lifecycle.py \
  tests/test_experiment_config.py \
  tests/test_process_verifier.py \
  tests/test_unattended_promotion_controller.py
```

当前公开测试集结果为 **38 passed**。GitHub Actions 会在 push 和 pull request 时执行同一组测试。

## GPU 训练环境

完整 GRPO 运行需要 Linux + NVIDIA CUDA。已验证的参考组合为：

| 组件 | 版本 |
| --- | --- |
| Python | 3.10 |
| PyTorch | 2.10.0 + cu128 |
| veRL | 0.8.0 |
| vLLM | 0.17.0 |
| GPU | NVIDIA A100 80 GB |

安装 GRPO extra 后，先做运行时检查，再仅查看解析后的训练命令：

```bash
pip install -e '.[grpo]'
PYTHONPATH=.:src python scripts/check_grpo_runtime.py
PYTHONPATH=.:src python scripts/train_grpo.py --dry-run
```

训练需要由使用者提供环境服务、数据路径和模型路径。`configs/grpo.yaml` 通过环境变量显式绑定这些外部资源，避免将机器路径或凭证硬编码进代码。请先阅读 [`docs/grpo.md`](docs/grpo.md) 与 [`docs/versioning.md`](docs/versioning.md)。

## 受控运行与失败恢复

GPU 实验不应覆盖旧目录或凭“进程退出”直接晋级。本项目的控制器使用如下状态流转：

```text
READY → RUNNING → COMPLETE / FAILED → ANALYZED → PROMOTED
```

- 运行前检查提交版本、数据/配置哈希、可执行文件、磁盘、端口与 GPU 资源；
- `COMPLETE` / `FAILED` 只能来自预注册 marker，不信任孤立的 exit code；
- 停滞 heartbeat 只冻结对应分支，避免错误推进；
- 新尝试必须使用新的 run ID，失败目录作为证据保留；
- `PROMOTED` 要求完成标记和全部预注册分析条件均通过。

示例 manifest 见 [`configs/unattended_promotion.example.json`](configs/unattended_promotion.example.json)，控制器入口为：

```bash
PYTHONPATH=.:src python scripts/unattended_promotion_controller.py --help
```

## 数据、环境与公开边界

为尊重上游许可和实验数据边界，本仓库**刻意不分发**：

- ShopSimulator 源码快照、商品目录、检索索引和私有 TaskFacts；
- 教师原始轨迹、训练/验证 JSONL 或 Parquet、模型权重与 checkpoint；
- 原始 rollout、运行日志、API 凭证、服务器地址和机器专属 launch receipt。

`results/` 只保留可读的聚合统计。若要复现完整流程，请从数据与环境权利方按其原始许可证和条款获取依赖；详细规则见 [`PUBLIC_DATA_POLICY.md`](PUBLIC_DATA_POLICY.md)。

## 贡献与开发原则

欢迎围绕可审计性和可复现性提交改进。提交前请确保：

1. 不提交 token、密钥、模型权重、环境数据或运行日志；
2. 新实验附带配置、输入哈希、代码提交与失败判定口径；
3. 不用隐藏目标、真实终局 Reward 细节或 LLM Judge 结果塑造过程奖励；
4. 新增行为应有公开可运行的测试或最小复现。

## 致谢

- [veRL](https://github.com/volcengine/verl)：在线 RL 与 Agent 训练框架；
- [vLLM](https://github.com/vllm-project/vllm)：高吞吐 rollout 推理；
- [ShopSimulator](https://github.com/ShopAgent-Team/ShopSimulator)：本项目评测所对接的上游购物 Agent 环境。

本仓库仅发布项目自有的适配、训练、评测与实验治理代码；不再分发上述上游环境或数据。

## License

[MIT](LICENSE)，仅适用于本仓库发布的项目自有代码；外部依赖、环境与数据仍遵循其各自许可证和使用条款。
