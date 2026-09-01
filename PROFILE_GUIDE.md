# BIRD-dev 全量成本测量 · 操作指南

> 给在新机器上接手这件事的人 / AI。**从头到尾照做即可,不需要读别的文档。**
> 唯一的入口脚本是 `run_profile_140g.sh`;它内部调 `run_test.sh`(四模块编排器)。

## 先看这里:你手上应该有什么

交付的是**整个 `LadderSQL_BIRD_submission/` 目录**(不是单个文件) —— 四个模块的代码都在里面。
其中与本次测量直接相关的是:

| 文件 | 作用 |
|---|---|
| **`PROFILE_GUIDE.md`** | 本文。**先把它读完**,尤其是 §2(机器要求)与 §3(并发口径) |
| **`run_profile_140g.sh`** | 唯一入口,一键跑完四个模块 |
| `run_test.sh` | 四模块串行编排器(由上面那个脚本调用,一般不直接跑) |
| `tools/profile_report.py` | 生成每题/每模块的时间 + token 报表 |
| `tools/merge_llm_usage.py` | 合并多进程 token 账本分片 |
| `tools/_smoke_multiproc_usage.py` | 开跑前的 5 秒自检:多进程 token 归属(见 §4.0) |
| `tools/_smoke_per_sample_report.py` | 开跑前的秒级自检:逐样本时间/token 报表 + 修复改写回查 |
| `tools/_smoke_judge_isolate.py` | 开跑前的秒级自检:模块 4 题间串行 / 题内并发口径 |

> 三个自检脚本已经内嵌在 `run_profile_140g.sh` 的第 1.2 / 1.3 / 1.4 步,不花钱、十几秒,
> 不通过就直接退出。**不需要你手动跑**,列在这里只是让你知道它们在做什么。

### ⚠️ 目录里有 5 个 `.sh`,只跑一个

其余四个是历史/其它用途的脚本,**跑错了不会报错,但数据全废** —— 它们不含本次的
并发与计量口径修正。对照表:

| 脚本 | 跑吗 | 说明 |
|---|---|---|
| `run_profile_140g.sh` | ✅ **就跑这个** | 本指南对应的唯一入口 |
| `run_test.sh` | ❌ | 四模块编排器,由上面那个调用。直接跑会缺一大截口径环境变量 |
| `run_dev_full.sh` | ❌ 千万不要 | 作者的**个人脚本**,硬编了一个**已失效**的 DashScope key,且不含本次任何修复。名字看着很像入口,实际跑完数据不可用 |
| `run_gen_dlc.sh` | ❌ | 旧的 DLC 上只跑生成模块的脚本。真需要分机跑用 `STAGES=gen bash run_profile_140g.sh` |
| `make_code_zip.sh` | ❌ | 打包榜单提交用,与成本测量无关 |

除代码以外还需要两个外部条件,**缺一不可**:
1. 挂载同一个 OSS 到 `/mnt/data`(权重/数据/venv 都在上面,见 §2.5)
2. 一个**当前有效的** GLM-5.2 API key(代码里硬编的那个已失效;现成可用的配置见 §2.3)

---

## 1. 要产出什么

在 BIRD **old-dev(1534 题)** 上完整跑一遍 LadderSQL 四模块链路,同时采集三件东西:

| # | 产出 | 验收标准 |
|---|---|---|
| 1 | 最终 EX | **约 73.66**(±0.5 内算对齐;偏离 >1 点说明配置被改动了,见 §7) |
| 2 | 每题 / 每模块 **端到端耗时** | 四个模块都要有非零数据,题数覆盖 = 1534 |
| 3 | 每题 / 每模块 **输入/输出 token 数** | 四个模块都要有非零 token |

第 2、3 项的**逐样本**明细落在 `logs/per_sample_cost.csv`，一题一行、共 18 列:

```
task_idx, db_id, question,
schema_link_time, schema_link_in, schema_link_out,
generation_time,  generation_in,  generation_out,
repair_time_amortized, repair_in, repair_out,
judge_time, judge_in, judge_out,
e2e_time, e2e_in, e2e_out           ← 端到端 = 四模块之和
```

> **端到端时间是四个模块的和,不是单一墙钟。** 因为四模块严格串行(模块 1 把
> 1534 题跑完才进模块 2),同一道题的四段耗时并不连续发生,不存在“这道题的
> 端到端墙钟”这个物理量。这是用户指定的测量口径,汇报时要写清楚。

产物均**实时落盘**(不是跑完才写):token 明细每次调用 append+flush;模块 1/4 的
每题墙钟每题 flush;模块 2 按 `--save-every` 增量存盘(原子替换)。中途中断不丢已完成的数据,
且重跑同一命令会断点续跑。`per_sample_cost.csv` 是跑完后由 `profile_report.py` 一次生成的 ——
想中途看阶段结果可以随时手动跑它(命令见 §5)。

四个模块的固定叫法(全文一致):

| 模块 | 名字 | 算力 | stage tag |
|---|---|---|---|
| 1 | schema link(APEX-SQL) | 闭源 API | `schema_link` |
| 2 | SQL 生成(自有 14B) | **GPU** | `generation` |
| 3 | SQL 修复 | 闭源 API | `repair` |
| 4 | SQL 选择(成对裁决 + 选择) | 闭源 API | `judge` |

**四个模块严格串行**:模块 1 把 1534 题全部跑完,才进模块 2。不是流水线。

---

## 2. 机器要求(硬性,缺一不可)

### 2.1 GPU:单卡显存 ≥ 60 GB(目标配置 140 GB)

为什么卡这条线 —— 模块 2 要求「一题的 32 个候选**同时**在生成」,否则测出的"32 并发"名不副实:

```
14B bf16 权重                        28 GB
KV cache 每 token                   192 KB   (48 层 × 8 KV heads × 128 head_dim × 2 × 2B)
32 条序列 × 约 4k token/条           ≈ 25 GB  KV 需求
```

| 卡 | util 0.90 后的 KV 池 | 可容 token | 32 路真并发? |
|---|---|---|---|
| L20 46 GB | 13.4 GB | 73 k | ❌ 只有 18–24,候选会分批排队 |
| A100 80 GB | 44 GB | 240 k | ✅ |
| **140 GB** | **98 GB** | **535 k** | ✅ 4 倍余量 |

脚本第 0.1 步会自动检查并在不足时**直接退出**并说明原因。
如果确认要在小卡上跑(仅为验证流程,或接受更低真并发),用 `MIN_GPU_MB=<实际显存>` 显式降低门槛 ——
但**报告里必须写明卡型与实际真并发数,不能声称 32 并发**。

> 注:`MAX_MODEL_LEN=24576` 是单序列上限,不是实际占用。真实上下文约 3–5k token,
> 所以 535 k 的池子对 32 条序列非常宽裕。**不要为了省显存去改 MAX_MODEL_LEN**,
> 它是复现 73.66 的配置之一。

### 2.2 公网出口(必须能连 `JUDGE_API_BASE` 指向的域名)

模块 1 / 3 / 4 **全部**依赖闭源 LLM(GLM-5.2)。只有模块 2 是纯本地 vLLM。

域名取决于你用哪家供应商 —— 见 §2.3,当前已实测可用的是 SiliconFlow
(`api.siliconflow.cn`)。DashScope 那套 key 已失效,不要再用。

脚本第 1.1 步会**真发一次 API 调用**验证。不通就退出 —— 不要跳过,否则会逐题 401
空转几小时、产出全是空结果。

> 如果新机器确实没有公网(例如某些训练集群的计算节点):
> 只能先跑模块 2(`STAGES=gen bash run_profile_140g.sh`),把候选池留在 `WORK_DIR`,
> 之后在有公网的机器上跑 `STAGES=api`。中间产物走共享存储,可以跨机接续。

### 2.3 API key 与额度 ⚠️ 这会真花钱

#### 已实测可用的配置(SiliconFlow) —— 直接用这套

```bash
export JUDGE_API_KEY=<你自己的 SiliconFlow key>   # 形如 sk-xxxxxxxx…，单独提供，不要写进任何文件
export JUDGE_API_BASE=https://api.siliconflow.cn/v1
# 模型名会由脚本自动推断成 zai-org/GLM-5.2, 不用手动设
```

**注意模型名带组织前缀**:SiliconFlow 上叫 `zai-org/GLM-5.2`,不是 `glm-5.2`。
`run_profile_140g.sh` 会根据 `JUDGE_API_BASE` 自动推断(含 `siliconflow` → 用带前缀的名字),
也可以用 `GLM_MODEL=<名字>` 显式覆盖。

实测结果(2026-08,本机):

| 项 | 结果 |
|---|---|
| 32 并发 · 真实负载(2372 token prompt) | **32/32 成功,零限流**,19.2 req/s,p95 延迟 1.5 s |
| 是否返回 `usage` | ✅ 返回(token 计量的前提) |
| `enable_thinking=False` 是否生效 | ✅ 生效。**不关的话输出 token 从 1 涨到 129(其中 128 是 reasoning)** |

> 之前 DashScope 上那个 key(`run_dev_full.sh:89` 硬编的)**已失效**,实测 `401 invalid_api_key`。
> 不要用它。若要换回 DashScope,base 用 `https://dashscope.aliyuncs.com/compatible-mode/v1`、
> 模型名用 `glm-5.2`,并自己验一遍:
>
> ```bash
> curl -sS -w "\nHTTP=%{http_code}\n" "$JUDGE_API_BASE/chat/completions" \
>   -H "Authorization: Bearer $JUDGE_API_KEY" -H "Content-Type: application/json" \
>   -d '{"model":"'"$GLM_MODEL"'","messages":[{"role":"user","content":"hi"}],"max_tokens":5}'
> ```
>
> 期望 `HTTP=200`。拿到 401 就是 key 问题,不是网络问题(401 能收到结构化错误响应,
> 恰好证明网络是通的)。`JUDGE_API_BASE` 末尾带不带 `/` 都可以,已实测两种写法等价。

跑一次全量 dev 的闭源 LLM 消耗(dev 实测值,模块 1 的量本次才第一次测):

```
模块 3 修复   约 5,244 次调用    ~6.2 M 输入 / ≤3.7 M 输出 token
模块 4 判别   约 14,952 次调用   ~38.1 M 输入 / ≤10.5 M 输出 token
模块 1 APEX   未知(本次要测)     预留 10–20 M 输入
────────────────────────────────────────────────────────
合计         预算 55–70 M 输入 / 15–20 M 输出 token
```

**开跑前先确认额度够**。中途额度耗尽会在断点停下(可续跑),但已花的 token 不退。

### 2.4 本地磁盘 ≥ 80 GB、内存 ≥ 64 GB、CPU ≥ 32 核

权重 28 GB + dev 库解压 + 候选池(约 0.5 GB)+ 各级缓存。32 进程并发需要核数。

### 2.5 资产必须可读:挂载同一个 OSS 到 `/mnt/data`

脚本从这些路径读(可用 `ASSETS=` 覆盖根目录):

| 资产 | 路径(相对 `/mnt/data`) | 大小 |
|---|---|---|
| 14B 权重(dev 73.66 的那个 checkpoint) | `training_outputs/ReAct_agent_bird_redundant/20260711_235128/checkpoints/global_step_192/actor/huggingface` | 28 G |
| dev 题目 + gold | `nl2sql_dataset/bird/dev_20240627/dev.json` | 724 K |
| dev 数据库 | `nl2sql_dataset/bird/dev_20240627/dev_databases.zip` | 330 M |
| 预建 SCHEMA_CACHE | `agl_20260626/apex-sql-schema-link-results/bird_old_dev/schema_cache_old_dev_rendered.json` | 4.9 M |
| APEX-SQL 源码 | `APEX-SQL-Project-main/APEX-SQL-Project-main/BIRD` | 36 M |
| 打包好的 venv(含 vllm/torch) | `venv_backup.tar.gz` + `uv_python_3.12.12.tar` | 5.7 G |

脚本第 0.3 步逐个检查,缺哪个就报哪个。

---

## 3. 并发与计时口径(理解这一节,才能看懂结果)

用户要求:**统一 32 并发**。落地时分两类 —— 判断依据是「一道题内部有没有天然的并行子任务」:

| 模块 | 开关 | 语义 | 每题时间的定义 |
|---|---|---|---|
| 1 schema link | `APEX_PROC_WORKERS=32` | 32 个**进程**跑样本队列,始终 32 题在飞,跑完一个立即补一个 | 该题 APEX 全流程墙钟 |
| 2 SQL 生成 | `ISOLATE_SAMPLES=1`<br>`PER_GPU=32` `MAX_NUM_SEQS=32` | 样本间**串行**;题内 32 候选并发 | 该题 32 候选从开始到全部返回 |
| 3 SQL 修复 | `--workers 32` | 32 个待修 SQL 并发 | ⚠️ 见 §8 已知限制 |
| 4 SQL 选择 | `ISOLATE_SAMPLES=1`<br>`JUDGE_SAMPLE_CONC=0`<br>`MULTIREP_WORKERS=32` | 样本间**串行**;该题的所有成对裁决并发(上限 32) | 该题全部裁决从开始到全部返回 |

模块 **2 和 4** 一道题内部就是一批互不依赖的子任务(32 个候选 / N 个成对裁决),
所以都是「样本间串行 + 题内并发」。模块 **1 和 3** 一道题只有一串顺序调用,题内无可并发,
所以走样本间并发。

### 三个设计决定,以及为什么

**(a) 模块 1 用进程而不是线程。**
APEX 内部有嵌套线程池(PLAN/PRUNE 采样)。token 记账靠 `llm_usage.set_unit()`,
它是 `threading.local` —— 嵌套的子线程读不到,只能回退到模块全局 `_GLOBAL_UNIT`。
一个线程池里 32 道题共享同一个模块全局变量,**token 会归到错误的题上**。
改成 32 个进程后,每个进程同一时刻只有一题,`_GLOBAL_UNIT` 无歧义。
每个子进程写自己的账本分片(`WORK_DIR/apex/usage_parts/`),跑完由
`tools/merge_llm_usage.py` 合并 —— 因为多进程 append 同一文件在 ossfs 上不保证原子,
而 `llm_usage.json` 的 read-merge-replace 会互相覆盖丢数据。

> 合并时的**汇总数字一律从明细行重建**,而不是读子进程的 `llm_usage.json`。
> 原因:子进程由 `multiprocessing.Pool` 管理,退出走 `os._exit()`,**不执行 `atexit`**,
> 所以 `llm_usage.dump()` 永远不会被触发、json 分片根本不存在。明细 JSONL 是每次
> 调用实时 append+flush 的,必然完整。这一点已由自检脚本验证(见 §4.0)。

**(b) 模块 2 的 `MAX_NUM_SEQS` 设成和客户端并发相等(都是 32)。**
这样一题的 32 个候选在 vLLM 内部**零排队**,每题墙钟不含等槽位的时间。
候选池里同时记两个字段,含义不同,别混用:
- `wall_sec` = 首个候选入队 → 最后一个返回(**含**等槽位)
- `wall_exec_max` = 32 个候选各自执行墙钟的**最大值**(**不含**等槽位)
样本间串行时两者应当接近;差得远说明并发被超订了。

> **成本报表用哪个?** 靠每题的 `isolate_samples` 标记自动选:
> 本次是隔离模式(样本间串行) → 取 **`wall_sec`**。理由:隔离下它已经不含跨样本
> 排队,而且比 `wall_exec_max` 多含了题内调度开销 —— 它才是这题真实的端到端
> 时间。若取 `wall_exec_max` 会系统性**低估**模块 2 与端到端总时间,而这正是要上报
> 给官方的数字,宁可保守也不能虚低。非隔离池则相反(排队会让 `wall_sec` 虚高),
> 退回 `wall_exec_max`。两条路径都被 `_smoke_per_sample_report.py` 的 \[7] 钉住。

**(c) 模块 4 的「32」是并发上限,不是 32 路真并发 —— 汇报时必须区分。**

一道题的裁决数由它自己的结构决定:

```
该题裁决数 = C(参赛簇数, 2) × (每簇写法数)² × 2      最后的 ×2 是 A/B 位置互换
```

| 簇数 | 每簇写法 | 该题裁决数 | 32 并发下的批次 |
|---|---|---|---|
| 2 | 1 | 2 | 1(只有 2 个在飞) |
| 2 | 3 | 18 | 1(只有 18 个在飞) |
| 3 | 2 | 24 | 1 |
| 3 | 3 | **54** | 2(第一批才打满 32) |

所以大多数题的实际在飞数 **小于 32**。这跟模块 2 不同 —— 模块 2 一题恰好 32 个候选,
名副其实。每题的真实在飞数已记在 `judge_verdicts_sample_wall.jsonl` 的 `calls` 字段里,
汇报时写「并发上限 32,逐题实际并发度见 `calls`」,不要笼统写「32 并发」。

> 另:题内并发前会先**串行预热**该题的执行结果与 preview 缓存。不预热的话,
> 题内几十个线程会同时撞上同一批 cache miss 、各自重复执行同一条 SQL,
> 把 sqlite 耗时混进判别墙钟里。这一点由 `tools/_smoke_judge_isolate.py` 验证(见 §4.0)。

### ⚠️ 三个并发开关的优先级（不要动）

`ISOLATE_SAMPLES=1` 同时作用于模块 2 和模块 4,这正是我们要的 —— 两者都该「样本间串行」。
但模块 1 不能串行(会慢几十倍),所以用 `APEX_PROC_WORKERS=32` **覆盖**它。

最容易踩的坑是 `JUDGE_SAMPLE_CONC`:

| 取值 | 模块 4 的实际口径 |
|---|---|
| **`0`(本次采用)** | 题间串行 + 题内并发 |
| 非 0 | **反过来接管**成题间并发 + 题内串行(优先于 `ISOLATE_SAMPLES`) |

它设错不会报错,只会静默地换掉每题墙钟的含义。`run_profile_140g.sh` 已硬写为 0,
并由 §4.0 的自检把守。**这三个变量都不要改。**

---

## 4. 怎么跑

### 4.0 先跑自检(十几秒,不花钱、不占 GPU)

`run_profile_140g.sh` 已经在第 1.2 / 1.3 / 1.4 步自动跑它们了,**不需要你手动执行**。
想单独验证时:

```bash
python tools/_smoke_multiproc_usage.py    # 多进程 token 归属
python tools/_smoke_per_sample_report.py  # 逐样本时间/token 报表
python tools/_smoke_judge_isolate.py      # 模块 4 的并发口径
```

三个分别把守一件事:

| 自检 | 验证什么 | 它堵的是哪个坑 |
|---|---|---|
| `_smoke_multiproc_usage` | 40 题 / 8 进程 / 每题 3 次**嵌套线程**调用 → token 不串题、合并后总量守恒、重复合并幂等 | §3(a) 那套多进程记账 |
| `_smoke_per_sample_report` | 四模块按题 join、CSV 18 列导出、被修复改写的 SQL 能否回查到 repair token、`isolate_samples` 两条取值路径 | 每题 repair token 漏计;模块 2 每题耗时被低估 |
| `_smoke_judge_isolate` | 模块 4 真的题间串行、题内并发打满上限、预热消除重复执行、`calls` 对得上、裁决缓存原子落盘且能断点续跑 | 跑错分支 → 每题墙钟口径错 |

都应看到最后一行 `结论: 全部通过`。**这三项是整份成本数据可信的前提**,任一失败就不要往下跑。

### 4.1 再跑试点(强烈建议,约 20–40 分钟)

```bash
cd <submission 目录>
export JUDGE_API_KEY=<glm-5.2 的 key>
LIMIT=30 bash run_profile_140g.sh
```

试点默认**均匀跨库抽样**(`PILOT_MODE=spread`):实测抽 30 题能跨 **11 个数据库**。
为什么不取前 30 题 —— BIRD dev 是按数据库排序的,前 30 题**只落在 1 个库**里,
用它估全量速率会有偏(单库的 schema 大小、缓存命中不代表平均)。
只想最快验证能不能跑通时用 `PILOT_MODE=head LIMIT=30 bash run_profile_140g.sh`。

试点四个模块全走一遍,产物格式与全量**完全一致**。
目的有三个:① 验证环境通了 ② 拿到真实速率来推算全量需要多久
③ 确认 token 记账四个模块都有数。

试点通过后,用它的速率估全量:`全量小时数 ≈ 试点秒数 / 30 × 1534 / 3600`。

### 4.2 全量

```bash
export JUDGE_API_KEY=<glm-5.2 的 key>
nohup bash run_profile_140g.sh > /dev/null 2>&1 &
tail -f <WORK_DIR>/logs/run.log
```

产物目录默认 `/mnt/data/agl_20260626/_profile_dev_full/`(试点是 `_profile_dev_pilot30/`)。
**所有产物都落在这个持久目录里** —— 机器释放后不会丢。

### 4.3 预计耗时(估算,以试点实测为准)

| 模块 | 单卡 140 GB 预计 |
|---|---|
| 1 schema link(32 进程) | 0.5 – 2 h(取决于限流) |
| 2 SQL 生成(样本串行) | **4 – 8 h** ← 最大不确定项 |
| 2c 执行缓存 | 10 – 20 min |
| 3 SQL 修复 | ~25 min |
| 4 SQL 选择 | 0.5 – 1.5 h |
| **合计** | **约 6 – 11 h** |

模块 2 的估算基础:总生成约 15 M 输出 token,32 路并发,`--enforce-eager`(无 CUDA graph)。
样本间串行会有每题的收尾拖尾(32 个候选不会同时结束),实际 GPU 利用率约 60%。
**这是估算,不是实测 —— 必须用试点校准。**

模块 4 在新口径(题间串行)下的耗时**不取决于总裁决数,而是题数 × 每题批次**:
约 620 道多簇题 × 1–2 批 × 单次延迟。因为题内能并发,绝大多数题只需 1 批,
所以它比「题间并发 + 题内串行」只慢约 1.5 倍,不是慢 N 倍。

---

## 5. 跑完怎么验收

```bash
W=/mnt/data/agl_20260626/_profile_dev_full
cat $W/logs/cost_report.txt          # 汇总: 每模块均值/极值 + 端到端
head -3 $W/logs/per_sample_cost.csv  # 逐样本: 一题一行, 18 列
cat $W/logs/ex_score.txt             # EX
```

逐条对:

- [ ] **`logs/soft_failures.txt` 不存在**。它存在就说明收尾三件事(格式校验 / 成本报表 /
      EX)里有失败项 —— 这三项不会中断流程(故意如此,避免一项挂掉连带弄丢另两项),
      所以**必须主动看这个文件**,不能只看脚本退出码。失败项可单独重跑,不需重跑四个模块。
- [ ] **EX ≈ 73.66**。偏离 >1 点 → 检查 §7 的「不能改的参数」是否被动过。
- [ ] `per_sample_cost.csv` **行数 = 1534**(试点时 = 试点题数)。
- [ ] `cost_report.txt` 里 end-to-end **JOINED coverage = 1534**。小于说明有题没 join 上
      (通常是 `db_id|||question` 这个 key 在某个模块里不一致)。
- [ ] 日志里有 `[keymap] 载入 N 条修复改写映射`。若看到 `未找到 ...repair_keymap.json`,
      则被修复改写过的候选其 repair token 会漏计(`coverage.repair` 会明显偏小)。
- [ ] 四个模块(`schema_link` / `generation` / `repair` / `judge`)**都有非零 token**。
      `schema_link` 为 0 是最常见的坑 → 说明 `usage_parts` 没合并,手动补:
      ```bash
      python tools/merge_llm_usage.py \
        --parts-dir  $W/apex/usage_parts \
        --out-detail $W/logs/llm_usage_detail.jsonl \
        --out-json   $W/logs/llm_usage.json
      python tools/profile_report.py --pool $W/candidates_test_32_repaired.json \
        --detail $W/logs/llm_usage_detail.jsonl \
        --stage-wall $W/logs/stage_wall.json \
        --judge-wall $W/cache/judge_verdicts_sample_wall.jsonl \
        --schema-wall $W/apex_out/schema_link_sample_wall.jsonl \
        --out-per-sample $W/logs/per_sample_cost.csv \
        --out $W/logs/cost_report.json
      ```
      上面这条 `profile_report.py` 也是**中途想看阶段成本**时的命令 —— 它只读已有产物,
      随时可跑、不影响正在进行的任务(keymap 会自动从 `<pool>.repair_keymap.json` 发现)。
- [ ] 每题墙钟文件行数 = 题数:
      ```bash
      wc -l $W/apex_out/schema_link_sample_wall.jsonl        # 模块 1
      wc -l $W/cache/judge_verdicts_sample_wall.jsonl        # 模块 4(只统计多簇题, 会少于 1534, 正常)
      ```
- [ ] 模块 2:候选池里 `wall_sec` 与 `wall_exec_max` 接近(差很多 = 并发被超订)。

---

## 6. 产物清单(要交回去的就是整个 `WORK_DIR`)

```
WORK_DIR/
├── predict_dev.json                     最终预测 {"0": "SELECT ...", ...}
├── candidates_test_32.json              模块 2 候选池(每题 32 条 + 每题墙钟 + token)
│                                       每题带 `isolate_samples` 标记: =true 时成本报表
│                                       取 `wall_sec`(真实端到端), 否则取 `wall_exec_max`
├── candidates_test_32_repaired.json      模块 3 修复后的池(成本报告读这个)
├── candidates_test_32_repaired.json.repair_keymap.json
│                                       修复改写映射(修后 SQL -> 修前写法)。
│                                       报表靠它把 repair token 归到题上, 丢了就漏计
├── apex_out/
│   ├── results_<model>.json              模块 1 的 schema linking 原始输出
│   │                                     (模型名里的 '/' 会换成 '_',
│   │                                      如 results_zai-org_GLM-5.2.json)
│   ├── schema_link_sample_wall.jsonl    模块 1 每题墙钟
│   └── failures.json                    失败题 + 限流统计
├── apex/usage_parts/                    模块 1 各子进程 token 账本分片(合并后变 .merged)
├── cache/
│   ├── exec_cache.json / exec_cache_repaired.json
│   ├── judge_verdicts.json              模块 4 裁决缓存(可续跑)
│   └── judge_verdicts_sample_wall.jsonl 模块 4 每题墙钟 + `calls`(该题裁决数)
└── logs/
    ├── run.log                          全量日志
    ├── per_sample_cost.csv              ★★ 逐样本明细: 一题一行、18 列
    │                                     (四模块各自的时间/in/out + 端到端)
    ├── cost_report.txt / .json          ★ 汇总: 每模块均值/极值 + 端到端
    ├── ex_score.txt                     ★ EX
    ├── llm_usage.json                   闭源 LLM 总账
    ├── llm_usage_detail.jsonl           ★ 逐次调用明细(成本报告的原料)
    ├── soft_failures.txt                仅当收尾步骤(格式校验/报表/EX)有失败时才出现
    │                                     每次运行重置; 没这个文件 = 收尾全通过
    └── stage_wall/{schema_link,generation,repair,judge}   各阶段总墙钟(秒)
```

---

## 7. 中断、续跑、以及绝对不能改的参数

### 断点续跑

**整条链路每个模块都可续跑。** 中断后重跑同一条命令即可,已完成的不会重做:

| 模块 | 续跑依据 |
|---|---|
| 1 | `apex/parts/<qid>.json` 每题一个分片 |
| 2 | 候选池里的 `task_idx` |
| 3 | `cache/repair_ckpt.json` |
| 4 | `cache/judge_verdicts.json` 按裁决 key 缓存 |

### 不能改的参数(改了就复现不出 73.66)

```
N_TRIALS=32          每题候选数
TEMPERATURE=1.1      采样温度
MAX_TURNS_AGENT=5    ReAct 最大轮数
MAX_MODEL_LEN=24576  上下文上限
MAX_TOKENS=3072      单次生成上限
VARIANT=biased  TOPK=3  RMAX=3  KPRIOR=1.02     选择阶段超参
SCHEMA_CACHE         必须用预建的那个文件(见下)
```

### ⚠️ 关于 SCHEMA_CACHE:这不是 bug

模块 1 会真跑一遍 APEX 并产出 `apex_out/results_glm-5.2.json`,
但**模块 2 用的是预建的、稳定的 `SCHEMA_CACHE`,不是本次 APEX 的新输出**。

这是故意的:
- 模块 1 跑一遍,**只为测它的时间和 token 成本**;
- EX 要严格对齐 73.66,就必须用当初那份 schema —— 因为 APEX 输出有随机性,
  而且从 APEX 原始输出到最终 schema 还有三步后处理(语义增强 + 值向量检索 + 渲染,
  见 `STAGE1_RUNBOOK.md`),其中值向量检索需要 ChromaDB + all-MiniLM-L6-v2,
  跳过它大约掉 3 个 EX 点。

所以:**不要试图把本次 APEX 的输出接到模块 2 上去**,那会同时改变 EX 和成本口径。

---

## 8. 已知限制(汇报结果时必须一并说明)

1. **模块 3 的「每题时间」是摊派值,不是实测。**
   修复模块会把 32 个候选**跨题去重**成唯一 SQL 写法再修 —— 它的天然单位是「写法」
   而不是「题」。`profile_report.py` 两种都报:per-writing(实测)和
   per-question(把每个写法的开销摊回包含它的那些题)。汇报时说清用的是哪个。

2. **模块 4 的「32 并发」是上限,不是实际并发度。**
   一题的裁决数 = `C(簇数,2) × 写法数² × 2`,实际在 2~54 之间,**大多数题不足 32**。
   逐题的真实在飞数在 `judge_verdicts_sample_wall.jsonl` 的 `calls` 字段。
   汇报句式:「题间串行,题内并发上限 32,逐题实际并发度中位数 = X」。见 §3(c)。

3. **模块 4 只覆盖多簇题**,单簇题不需要裁决,所以它的每题墙钟文件行数少于 1534,
   这是正常的,不是漏数据。`per_sample_cost.csv` 里这些题的 `judge_*` 为 0。

4. **`repair_time_amortized` 是摊派值**(阶段总墙钟 ÷ 题数),因为修复是跳题去重后
   按「SQL 写法」批处理的,没有每题墙钟这个物理量。但同一行的 `repair_in/out` 是
   **真实之和**(本题各写法的 token 相加),不是摊派的 —— 两个字段的口径不同,别弄混。
   另:这些 token 能归到题上依赖 `repair_keymap.json`(账本按修复**前**的写法记账,
   而候选池存的是修复**后**的 SQL)。该文件丢失 → 被改写过的候选 token 漏计,
   表现为 `coverage.repair` 明显小于题数。

5. **模块 2 的耗时强依赖卡型**。报告里必须写明卡型和显存,不能只写「32 并发」。

6. **`--enforce-eager` 是开着的**(无 CUDA graph),生成速度损失约 10–20%。
   这是当初跑出 73.66 时的配置,为了口径一致没有关掉。

7. **模块 2 的每题耗时取 `wall_sec`(隔离模式下),包含题内调度开销。**
   它比 `wall_exec_max` 略大,是有意选的保守口径(见 §3b)。若对方要求「纯模型
   执行时间」,用候选池里的 `wall_exec_max` 另行统计,两个数都在池里。

8. **断点续跑时,那些「已调过 LLM 但未落盘就被中断」的题会被重跑。**
   它们的 token 在明细里会出现两次 —— 作为「实际花钱」是对的,作为「每题成本」
   会偏高。中断次数少时影响可忽略;若中断很多次,在报告里说明这一点。

---

## 9. 排错表

| 现象 | 原因 | 处理 |
|---|---|---|
| 启动即 `[FATAL] 显存 ... < ...` | 卡小于 60 GB | 换卡;或 `MIN_GPU_MB=<实际显存>` 强行跑并在报告里写明真并发数 |
| `401 invalid_api_key` | key 无效/过期/被轮换 | 换一个有效 key。注:能收到 401 说明**网络是通的**,不要往网络方向排 |
| `[FATAL] glm-5.2 preflight 失败` | 无公网 / key 无效 / 额度耗尽 | 连接超时→无公网;401→key 问题。见 §2.2、§2.3 |
| 无公网但想先跑生成 | — | `STAGES=gen bash run_profile_140g.sh`(已验证会自动跳过 key 与公网检查) |
| 模块 4 日志里看到 `[sampleq]` 而不是 `[isolate]` | `JUDGE_SAMPLE_CONC` 被设成了非 0 | 口径被静默翻成了「题间并发 + 题内串行」。设回 0 重跑模块 4(裁决有缓存,不会重复花钱)。见 §3(c) |
| 模块 4 日志有 `[isolate][WARN] 题内并发上限只有 N` | `MULTIREP_WORKERS` 小于 32 | `export MULTIREP_WORKERS=32` 后重跑模块 4 |
| 模块 1 大量 `401` | `OPENAI_API_KEY` 没同步 | 脚本已自动同步;若手工跑请一起 export |
| 模块 1 大量 `429` / 限流 | 32 进程打太猛 | 脚本会自动退避重试;仍严重就降 `CONC=16` |
| `cost_report` 里 `schema_link` token = 0 | `usage_parts` 未合并 | 跑 §5 里的 `merge_llm_usage.py` |
| JOINED coverage < 1534 | `db_id\|\|\|question` key 不一致 | 检查是否换过 `dev.json`(空白字符差异也会导致) |
| vLLM 起不来 / OOM | `GPU_MEM_UTIL` 太高或有残留进程 | `pkill -f "vllm serve"` 后重跑;或 `GPU_MEM_UTIL=0.85` |
| sqlite 随机 I/O error | 库放在 OSS 上直读 | 必须解压到本地盘(脚本已自动做) |
| 日志半天不动 | ossfs 上开着的文件不实时 flush | 看 `apex/parts/` 的文件数增长,那才是可靠进度信号 |
| 生成阶段 `wall_sec` 远大于 `wall_exec_max` | 并发超订 | 确认 `MAX_NUM_SEQS` = `PER_GPU` = 32 |
| 收尾打印 `[WARN] 以下收尾步骤未通过` | 格式校验 / 成本报表 / EX 中某一项失败 | 三者互不依赖且不会中断流程。中间产物都在,按 `logs/soft_failures.txt` 里的名字单独重跑即可(不需重跑四个模块) |
| `cost_report` 报 JSON 解析错 | 旧版本会把空的 stage_wall 分片拼成非法 JSON | 已修(空值与非数字自动跳过);若仍出现,直接删掉 `logs/stage_wall/` 里异常的那个文件重跑报表 |
| `per_sample_cost.csv` 里 `generation_time` 普遍偏小 | 候选池缺 `isolate_samples` 标记(旧池) → 退回了 `wall_exec_max` | 新跑的池子会自带该标记;旧池只能手动给每题补 `"isolate_samples": true` 后重跑报表。见 §3b |

---

## 10. 跑完要回报什么

把这几样交回去(或直接交整个 `WORK_DIR`):

1. `logs/cost_report.txt` 全文
2. `logs/ex_score.txt` 里的 EX 数字
3. **卡型 + 显存**(例:1 × H200 141 GB),以及总墙钟
4. 四个模块各自的**总墙钟**(`logs/stage_wall/` 四个文件)
5. 实际消耗的闭源 LLM token 总量(`logs/llm_usage.json` 的 `_total`)
6. §5 验收清单里**没通过的项**,以及 §8 里适用于本次的限制说明

---

## 附:改动过哪些代码(便于交叉核对)

为支撑本次测量,在原提交包上做了这些改动:

| 文件 | 改动 |
|---|---|
| `inference_selection/llm_usage.py` | 新增 `set_paths()`:多进程各自记账,避免明细串行与总账互相覆盖 |
| `schema_construction/apex_sql_runner/run_apex_sl.py` | 新增 `APEX_PROC_WORKERS`:进程级样本队列 + 每题墙钟/账本分片合并 |
| `inference_selection/tournament_blind.py` | ① 新增 `JUDGE_SAMPLE_CONC`(题间并发口径,本次置 0 不用);② 隔离模式改为本次默认口径(题间串行 + 题内并发),并写每题墙钟 + `calls`;③ 题内并发前串行预热 exec/preview 缓存(避免重复执行污染墙钟);④ 裁决缓存改原子写(tmp+fsync+replace),避免中途被杀写出半个 json 导致续跑全丢 |
| `inference_selection/run_gated_reasoning_judge.py` | 先剥模型名的组织前缀再判断厂商:否则 `zai-org/GLM-5.2` 会让 `startswith("glm")` 落空 → 不发 `enable_thinking` → 输出 token 百倍暴涨且结论被截断 |
| `sql_repair/repair_candidates_generic.py` | 落 `<out_file>.repair_keymap.json`(修后 SQL → 修前写法):账本按修前写法记账而池存修后 SQL,不留这张映射则每题 repair token 漏计 |
| `generation/sample_bird_candidates.py` | 每题输出新增 `isolate_samples` 标记:成本报表靠它判定该取 `wall_sec` 还是 `wall_exec_max`。缺这个标记会系统性低估模块 2 与端到端总时间 |
| `tools/profile_report.py` | ① 新增 `--out-per-sample`(逐样本 18 列 CSV) 与 `--repair-keymap`(回查被改写候选的 token);② 每题生成耗时改为根据 `isolate_samples` 选字段 —— 隔离模式取 `wall_sec`(真实端到端),旧池仍退回 `wall_exec_max` |
| `tools/merge_llm_usage.py` | 新增:合并多进程账本分片(幂等);汇总由明细行重建,规避 Pool worker 不跑 `atexit` 的坑 |
| `tools/_smoke_multiproc_usage.py` | 新增:多进程 token 归属自检(见 §4.0),已实测全部通过 |
| `tools/_smoke_per_sample_report.py` | 新增:逐样本报表自检(7 项,含 keymap 反向验证、以及 `isolate_samples` 两条取值路径),已实测全部通过 |
| `tools/_smoke_judge_isolate.py` | 新增:模块 4 并发口径自检(10 项,含题间串行/题内并发/预热去重/断点续跑),已实测全部通过 |
| `run_test.sh` | ① 模块 1 后自动合并账本;② 末尾新增 `SCORE_GOLD` EX 复核;③ 模型名不写死(含 `/` 时自动 slug 化);④ profile_report 调用加 `--out-per-sample`;⑤ 收尾三件事(格式校验/报表/EX)改用 `run_soft` 软失败 —— 它们互不依赖,旧逻辑下任一项挂掉会把另两项一起弄丢(典型后果:格式校验失败 → 几小时跑出的成本报表根本不生成);⑥ `stage_wall.json` 拼装跳过空/非数字分片,避免产出非法 JSON 废掉整份报表 |
| `run_profile_140g.sh` | 新增:本指南对应的一键入口(前置检查 + 环境还原 + 口径配置 + 3 项自检);按 API base 自动推断模型名;`MIN_GPU_MB` 可调门槛;`STAGES=gen` 自动跳过 API 检查;试点默认跳库抽样 |

这些改动**只影响计量与调度,不改变任何推理逻辑或超参**,因此不影响 EX。
唯一靠近推理的是 `run_gated_reasoning_judge.py` 那一处 —— 但它修的是「带前缀的模型名会让
`enable_thinking` 发不出去」这个 bug,修完才能回到与 73.66 相同的调用参数。
