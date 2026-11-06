# Terminal-Bench：本地 CPU/GPU 资源研究

本目录接入固定版本的 `terminal-bench/terminal-bench-cpu@4.1.0`，共 63 个任务。
CPU-only 指任务环境不需要 GPU；Qwen 推理仍使用 GPU。GAIA 与本目录独立，已有
数据、runner、框架注册和服务配置均不改动。

## 已提供的内容

| 文件 | 用途 |
| --- | --- |
| `configs/dataset.lock.json` | 官方 CPU-only 数据集内容哈希及全部 63 个任务的不可变引用 |
| `configs/suites.json` | 5 题环境检查、12 题候选试验清单 |
| `configs/qwen_tp2.yaml` | 现有 Qwen3.8-27B BF16、TP=2、128k context 的独立配置 |
| `run.py` | 生成 Harbor 计划；只有加 `--execute` 才运行 |
| `agents.py` | 项目 LangGraph ReAct 的 terminal 适配 |
| `budget_proxy.py` | 每个 trial 独立的调用/输出预算、原始请求和响应记录 |
| `export_results.py` | 原始 verifier 结果、终止状态及缺失 trial 的 CSV 导出 |
| `preflight.py` | 依赖、Docker、模型服务的只读检查 |
| `fixtures/terminal-smoke/` | 本地诊断任务，不计入 Terminal-Bench 成绩 |
| `tests/` | 无需 Docker/GPU/外部模型的集成测试 |

`data/`、`output/` 和本目录的 `.venv/` 被忽略，不提交任务内容和运行轨迹。
正式任务由 Harbor 在执行时下载，容器镜像沿用官方任务定义。解析后的数据集引用
和任务引用还会写入 Harbor 的 job 配置。

## 1. 环境

以下命令均在仓库根目录 `agent_tokenomic` 执行。Python >=3.12，Docker Engine
及 Compose 插件必须可用。不要给本项目安装云端运行环境或修改 Docker socket 权限。

```bash
bash terminal_bench/scripts/setup.sh
# 如果 docker compose/buildx 不存在，安装项目私有插件（不需要 sudo）：
python3 terminal_bench/scripts/setup_docker_plugins.py
terminal_bench/.venv/bin/python terminal_bench/preflight.py --skip-model
```

依赖安装在 `terminal_bench/.venv`，不会更新原 `.venv` 中的 vLLM 或框架。
`requirements.lock.txt` 是实际验证环境的完整版本锁。
插件 helper 固定官方 Compose/Buildx 版本和 SHA-256，仅支持 Linux x86_64，下载到
`terminal_bench/.docker/`。runner/preflight 在没有显式 `DOCKER_CONFIG` 时使用该目录；
不会修改全局 Docker 配置、守护进程或现有容器。该本地配置不携带私有 registry 凭据。

## 2. 先做无需模型的容器/评分检查

```bash
terminal_bench/.venv/bin/python terminal_bench/run.py \
  --agent oracle --local-tasks terminal_bench/fixtures \
  --task terminal-smoke --attempts 1 --execute
```

这会构建一个很小的 Python 容器，运行参考解法，再检查 `/app/result.txt` 是否为
正确结果。它可能拉取基础镜像，不使用 Qwen。通过不代表官方任务已经通过验证。

运行官方 5 题参考解法：

```bash
# 默认只预览，显示 5 题 × 5 次的计划，不下载任务或运行容器。
terminal_bench/.venv/bin/python terminal_bench/run.py --agent oracle --suite smoke

# 确认环境和所需时间后执行。官方任务可能较长；沿用其 reference/verifier timeout。
terminal_bench/.venv/bin/python terminal_bench/run.py \
  --agent oracle --suite smoke --attempts 5 --execute
```

默认并发为 1，无自动 trial 重试。每次生成唯一 job 名；显式同名计划或结果不会覆盖。
参考解法用于验证环境，不用于调节模型提示、挑选模型已通过的题，或向模型提供答案。

## 3. 接入本地 Qwen

先检查已有服务。此集成默认使用 **18000**，避免占用现有的 8000。
启动前确认 GPU 和端口可用；不会自动停止或替换其他项目的模型服务。

```bash
# 在另一个终端启动。沿用本项目的离线权重、Qwen parser 和服务记录方式。
PORT=18000 env -u ATK_STACK bash scripts/serve_vllm.sh \
  --config terminal_bench/configs/qwen_tp2.yaml

terminal_bench/.venv/bin/python terminal_bench/preflight.py
```

如果已经在其他本机端口运行相同的 Qwen，可通过 `--base-url http://127.0.0.1:PORT/v1`
复用；不要连到原本要求 `/control/run` 绑定的共享单 trial 代理。本 adapter 自带独立代理。
预检核对模型 ID，不能单独证明服务端 TP、精度和 thinking 设置；使用本配置启动并保存
`output/serving_<port>.json`，让实际服务配置进入实验记录。私有认证通过 `TB_API_KEY`
环境变量提供，配置内保留 `api_key: EMPTY`。

先用模型完成本地诊断任务：

```bash
terminal_bench/.venv/bin/python terminal_bench/run.py \
  --agent react --local-tasks terminal_bench/fixtures \
  --task terminal-smoke --attempts 1 --agent-timeout 60 --execute
```

然后预览或执行正式候选试验：

```bash
terminal_bench/.venv/bin/python terminal_bench/run.py --agent react --suite pilot

# 12 题 × 3 次；会启动正式模型实验。
terminal_bench/.venv/bin/python terminal_bench/run.py \
  --agent react --suite pilot --attempts 3 --execute
```

ReAct 复用 `agent/langGraph/react_agent.py` 的模型构造和现有 recursion budget 映射，
使用相同的 `langgraph.prebuilt.create_react_agent`。系统提示和工具换成 terminal 任务
所需内容，**不是逐字复用 GAIA 的 prompt/tools**。命令通过 Harbor `environment.exec`
进入任务容器；不会执行在宿主机。每条命令是新 shell，工作目录需要显式指定。
首版仅支持 terminal；遇到任务声明 MCP、skills 或会话恢复会明确拒绝，避免静默漏掉能力。

本地研究默认预算：200 次 LLM 请求、16,384 completion tokens/次、262,144 completion
tokens/trial、1 小时 agent timeout。thinking tokens 包含在 completion 预算中；HTTP
失败也消耗调用次数，禁止模型客户端重试。剩余 total budget 会降低下一次请求上限，
usage 缺失则禁止继续调用并标记 accounting_error。达到上限与因上限终止分别记录。
本适配使用非 streaming 请求；不报告伪造的 TTFT/TPOT。预算和计时属于本地资源研究
协议，不是官方排行榜默认协议。

## 4. 结果、逐步记录与评分

```text
terminal_bench/output/
├── plans/<job>.json             计划、任务引用、预算与部署类型
├── plans/<job>.cell.json        不含真实密钥的模型配置快照
└── jobs/<job>/<trial>/
    ├── result.json              Harbor 原始评分、错误、阶段时间和 usage
    ├── verifier/                原始 verifier 日志/reward
    └── agent/
        ├── trajectory.json     成功完成的 graph 消息序列
        ├── terminal_calls.json 完整命令、stdout/stderr、退出码、调用 wall time
        ├── accounting.json     调用/输出预算及终止口径
        ├── final_answer.txt
        └── raw/
            ├── 001_request.json / 001_response.json
            ├── llm_calls.json
            └── denied_requests.json  有被预算阻止的请求时生成
```

模型看到的 stdout/stderr 每项最多 24,000 字符，完整输出保存在 tool log。
发生 cap、错误或 timeout 时，`trajectory.json` 可能不存在；逐步原始请求/响应及工具日志
仍是审计来源。超时取消可能先于上游请求完成，usage 会标记不完整，不能把未知用量当 0。

```bash
terminal_bench/.venv/bin/python terminal_bench/export_results.py \
  terminal_bench/output/jobs/<job>
```

导出 `<job>.csv` 和 `<job>.quality.json`。仅接受指定 reward key 下的有限二元值；缺失、
非二元、verifier 异常和未产生 result.json 的计划运行都保留。导出区分：

- `verifier_pass`：官方 verifier 是否通过，以及 `reward_valid` 是否可用于评分。
- `completion_type`：自然完成、forced_final、step_limit、truncated、timeout、error 等。
- `strict_success`：通过 verifier，且已观测到自然完成，无运行异常。
- `verifier_pass_rate_all_planned`：以全部计划尝试为分母；另列 valid-only 分母，不能混用。

终止或 tokens 没有观测到时留空，不把未知值变成零或自然完成。不会额外调用 semantic
judge。官方 verifier 报告能力成绩，本地 strict 指标补充终止质量，两者同时保留。
ReAct 内置步数保护可能把模型的工具请求替换成一句结束语；此处对照原始响应检测，
将它标为 forced_final，不因 graph 返回了最终消息就计为自然完成。

## 5. CPU 与并发实验

```bash
terminal_bench/.venv/bin/python terminal_bench/run.py \
  --agent react --suite pilot --task-cpus 4 --concurrency 2
```

`--task-cpus` 是任务容器 CPU quota，不等于物理核心绑定，不控制宿主机上的 ReAct、
预算代理或 vLLM 的 CPU 核数。多容器任务/独立 verifier 的其他服务也可能有独立预算。
比较不同任务时保留其官方资源默认值；做资源干预时固定同一题，另行记录各 CPU pool、
NUMA、内存、线程数和主机上其他负载。修改 quota 后的结果须标为本地研究配置。

`llm_request_ms` 包括服务排队、CPU 和 GPU；`terminal_wall_ms` 包括 exec 传输和任务工作，
都不是纯 GPU/CPU 时间。并发试验应使用完整批次窗口计算吞吐和资源成本，失败运行也计入。
有需要时另起原有环境中的 `scripts/power_sampler.py --out terminal_bench/output/power.jsonl`；
该采样覆盖整机/设备，应独占资源或记录其他负载。不要把它归因到重叠运行的单一 trial。
本目录未把进程 CPU、GPU 能量或美元成本伪装成已测指标。

## 6. Hermes 与 NemoClaw 的边界

提供 `--agent hermes-native` 配置入口，固定已核对存在的 Hermes tag `v2026.9.24`，在 **Harbor 任务环境**
安装和运行原生 Hermes。必须显式给 `--container-base-url`，因为容器里的 localhost
不是宿主机。只应使用明确可达且获准访问的私有 bridge 地址；Docker/OpenShell 的
`host.*.internal` 别名不保证在另一个运行环境存在。

```bash
# 仅预览；先验证实际 task 容器到该地址的网络，再执行。
terminal_bench/.venv/bin/python terminal_bench/run.py \
  --agent hermes-native --task html-js-filter --attempts 1 \
  --container-base-url http://<private-bridge-ip>:18000/v1
```

**它不经过现有 NemoClaw/OpenShell 沙箱，也不测量其开销。** 原生 Hermes 的 max_turns
不等于代理强制的调用/输出预算，其计划会标记 `budget_enforced=false`。本地 cost/质量
结果不得直接宣称与 ReAct 预算公平；应先接同等 per-trial 代理、固定 sampling 和内部调用
统计，再开展正式双 agent 比较。`max_turns` 不限制所有辅助调用。要测试 NemoClaw，后续
需要让其工具操作 Harbor 的实际任务环境并保留隔离/策略边界，不能只复制题目到现有沙箱。

本次验证覆盖独立依赖、固定数据集元数据解析、配置预览、mock 模型的完整 ReAct 工具循环、
预算和导出；另已完成本地 Docker oracle/verifier 诊断任务，1/1 通过。真实 Qwen、官方任务
与原生 Hermes 尚未运行，不能把本地诊断当作这些正式任务的成绩。

## 离线检查

```bash
terminal_bench/.venv/bin/python -m unittest discover -s terminal_bench/tests -v
```

参考：[4.1 发布说明](https://github.com/harbor-framework/terminal-bench/releases/tag/v4.1.0)、
[官方运行指南](https://github.com/harbor-framework/terminal-bench)、
[Harbor custom agents](https://docs.harborframework.com/agents/custom-agents)。
