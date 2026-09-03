# Phase 0 — 先把不确定的事问清楚

这个目录里的东西**不参与业务**。它是一个只会睡觉、回话和 curl 的容器，
用来在动真代码之前回答四个问题。四个问题都查过官方文档，**一个都查不到答案**，
只能实测。

| # | 问题 | 为什么它能改变设计 |
| :-- | :--- | :--- |
| 1 | **一个请求最多能挂多久？** | 稽核要跑几十分钟。挂得住 → 流式；挂不住 → 轮询；根本不支持 → 跑完主动通知 |
| 2 | GE 能不能挂一个 Agent Runtime 上的自定义 agent | 挂不上，整条链路就不成立 |
| 3 | GE 支不支持中途等用户确认 | 「先确认再稽核」这条需求全压在它上面 |
| 4 | 容器能不能出网 | 出不去就没有视频可录，要先配 PSC interface 或 Agent Gateway |

---

## 结论（2026-09-03 全部跑完，细节和证据在下面）

| # | 答案 |
| :-- | :--- |
| 1 | **GE 那条路 602s，`:query` 896s。** 事先推断的「约 1 小时」**错了**。流式**更短**不是更长。而且流式**容器里的活 900s 会被 cancel**（unary 不会） |
| 2 | ✅ **能。** `agents.create` + `provisionedReasoningEngine`，走 A2A 真的调通了 |
| 3 | ✅ **能。** `contextId` 串两轮，`session_id` 原样回传容器 |
| 4 | ✅ **通。** 不用配 PSC，也不用 Agent Gateway |

**外加一条计划里完全没预料到、但影响最大的发现：**

> **GE 只会调一个方法**（`streaming_agent_run_with_events`），
> 把整轮对话原文丢进来。它**看不见** `classMethods` 里声明的其它方法，
> `description`/`parameters` 在这条路上一点用都没有。
> → 「三个方法给 GE 当三个工具挑」的设想不成立，**分流得在容器里自己做**。

**对 Phase 1/2 的三条硬要求：**

1. 每次 GE 往返必须 **10 分钟内**返回。
2. 稽核**不能 `await` 在 GE 那个请求里**——900 秒会被 cancel。
   甩进脱离请求的后台任务，立刻返回 job_id。
3. 状态必须外置，`session_id` 当主键。

目录里有什么：

| 文件 | 干什么 |
| :--- | :--- |
| `probe.py` / `Dockerfile` | 探针容器本体 |
| `measure.sh` | 打 Agent Runtime，测超时并出表 |
| `create_engine.py` | 建 / 改 / 删 reasoningEngine |
| `create_ge_app.py` | 建 GE 应用、注册 agent |
| `a2a.py` | **打我们自己的 GE agent**（`message:send`）。多轮确认和 GE 侧超时都用它 |
| `assist.py` | 打 `streamAssist`。**留着是因为它路由不到我们的 agent，这个差别本身是结论** |
| `results/` | 上面那些跑出来的原始日志和 JSON |
依赖只有 fastapi / uvicorn / aiohttp——**故意不装 Chromium 和 ffmpeg**，
这样探针要是出问题，原因一定在平台，不在我们。

---

## 探针提供的方法

`POST /api/reasoning_engine`，body 是 `{"class_method": ..., "input": {...}}`。
路由名和 body 结构都来自 runtime contract，不是随便起的。

| 方法 | 作用 |
| :--- | :--- |
| `hello` | 最小往返。确认部署可达 |
| `hang` | 睡 `seconds` 秒再回。**问题 1 的主力** |
| `probe_log` | 读回服务端自己的记录。**见下面「为什么要两步」** |
| `confirm_flow` | 两轮握手：第一轮抛问题，第二轮收答复。**问题 3** |
| `egress` | 从容器里访问外网。**问题 4** |

另外有一个**请求日志中间件**，把每一个进来的请求（包括 404 的）都记进 `probe_log`。
它是为了看清 `asyncQuery` 到底打哪条路由——见下面「查 schema 挖到的东西」。

`POST /api/stream_reasoning_engine` 只有一个 `stream_query`：每 `interval` 秒发一次心跳。
它有两个用途——contract 说 Console 的 playground 没有 `stream_query` 就打不开，
必须有；另外，**一直在吐字节的流和一个安静挂着的请求是两回事**，
超时机制通常只惩罚沉默。要是流能活 30 分钟而 `hang` 5 分钟就死，那就直接用流式推进度，
轮询那套备选方案根本不用做。

### 为什么测超时要分两步

客户端等不到回应时，它只知道**它自己不等了**。至于是容器被杀了、
还是连接被中间层掐了而活还在干、还是代理单方面挂断——从客户端看完全一样。

所以服务端把自己干完的事记在内存里，事后用 `probe_log` 读回来：

| 读到什么 | 说明 | 影响 |
| :--- | :--- | :--- |
| `hang_completed`，instance 没变 | 只是连接断了，**活还干完了** | 「跑完主动通知」这条路可行 |
| `hang_cancelled` | 平台把 cancel 传下来了 | 要处理中断和续跑 |
| 没记录，或 instance 变了 | 容器本身被换掉了 | 状态必须外置，不能留在内存 |

本地已经验过这个机制确实能区分：客户端 4 秒放弃一个 15 秒的 `hang`，
服务端照样在第 15 秒记下了 `hang_completed`，instance 没变。

---

## 查 schema 挖到的东西（散文文档上没有）

写探针的过程中去翻了 Vertex 的 API discovery document：

```bash
curl -s "https://aiplatform.googleapis.com/\$discovery/rest?version=v1" -o /tmp/disc.json
```

它是权威的，而且比文档页完整得多。挖到四条，**每一条都改变了判断**：

1. **`reasoningEngines:asyncQuery` 存在，而且明确支持 BYOC。**
   返回一个 `GoogleLongrunningOperation`，输入从 `inputGcsUri` 读、结果写 `outputGcsUri`，
   配套有 `cancelAsyncQuery` 和 `operations.get/wait/cancel`。
   原文：*"For BYOC, the content of the file depends on the agent application."*
   → **长任务是平台原生支持的**，「跑完主动通知」不是土办法。
2. **`keepAliveProbe.maxSeconds` 上限 3600 秒**（*"Can be a maximum of 3600
   seconds (1 hour)"*）。这是整个 API 表面上唯一一个明确的时长数字。
3. **`resourceLimits` 的说明直接链到 Cloud Run 的文档。** 底层是 Cloud Run 的话，
   请求超时上限就是它的 60 分钟。和第 2 条指向同一个天花板：**约 1 小时**。
   → **⚠️ 这个推断实测是错的，真实上限 896s / 602s。留着原文是为了记住怎么错的，
   复盘见下面「回头看」那一节。**
4. **`minInstances` 范围是 `[0, 75]`**，不是文档页写的 10。

第 2、3 条只是旁证，**不是实测**。Phase 0 要做的就是把这个推断钉死或推翻。
另外 runtime contract 只写了两条路由，`asyncQuery` 打到 BYOC 容器上走哪条路
完全没说——所以探针加了请求日志中间件，跑一次 `asyncQuery` 就知道了。

---

## 部署踩过的坑（**已解决，探针现在跑在云上**）

补完下面那条 IAM 授权之后，`containerSpec` 一次就通了。留着这一节是因为
**这条授权在客户环境同样要做**，是 BYOC 的固定前置步骤，不是我们项目的偶发问题。

现在活着的资源：

| 资源 | 名字 | 干什么 |
| :--- | :--- | :--- |
| reasoningEngine A | `projects/596821501265/locations/us-central1/reasoningEngines/645797604818419712` | 超时测量专用，`measure.sh` 打它 |
| reasoningEngine B | `.../reasoningEngines/2184621302495576064` | GE 实验专用 |
| GE 应用 | `projects/596821501265/locations/global/collections/default_collection/engines/cctv-audit` | |
| GE agent | `.../assistants/default_assistant/agents/16091433261218097511` → 引擎 B | 显示名「CCTV 稽核探针」 |

**为什么是两个引擎**：超时测量一跑一个多小时，中途重新部署会把容器换掉，
测出来的数字就废了。同一个镜像多建一个引擎不要钱，两组实验互不干扰。
`ENGINE_DISPLAY_NAME` 就是为这个加的。

### 原来卡在哪：服务代理读不了我们的 Artifact Registry，缺一条 IAM 授权

镜像推到 `us-central1-docker.pkg.dev/.../phase0-probe:v1` 之后，试了三次，全失败：

| # | 方式 | location | 耗时 | 结果 |
| :-- | :--- | :--- | ---: | :--- |
| 1 | `containerSpec` | `global` | 77s | code 13，只有一句「参考排障页」 |
| 2 | `containerSpec` | `us-central1` | 9s | code 3，`failed to start and cannot serve traffic` |
| 3 | `containerSpec` + `spec.serviceAccount` | `us-central1` | 9s | code 3，**这次报了真正的原因**（见下） |
| 4 | `sourceCodeSpec`（源码归档，平台自己构建） | `global` | 9min | code 13，同样只有一句「参考排障页」 |

前三次 **Cloud Logging 里一行容器日志都没有**，说明容器根本没起来。
（顺带确认日志管道本身是通的：这个项目 9 月 1 日部过一个
`investigation-agent`，`aiplatform.googleapis.com%2Freasoning_engine_stderr`
里有它完整的 uvicorn 启动日志。）

第 3 次开始平台给了人话——**加上 `spec.serviceAccount` 之后错误信息才变详细**，
这本身是个排障技巧：

> The Reasoning Engine could not access the container image referenced by
> `spec.container_spec.image_uri`. Ensure the image URI is correct and that
> \[AI Platform Reasoning Engine Service Agent] has permission to read it
> (**grant the `roles/artifactregistry.reader` role on the image's repository**).

和事先从 IAM 里推出来的一致。拉镜像的身份是 Reasoning Engine Service Agent
（`service-<项目号>@gcp-sa-aiplatform-re.iam.gserviceaccount.com`），
它挂的角色 `roles/aiplatform.reasoningEngineServiceAgent` 里：

```bash
gcloud iam roles describe roles/aiplatform.reasoningEngineServiceAgent \
  --format='value(includedPermissions)' | tr ',' '\n' | grep -iE 'artifact|storage'
# storage.buckets.get / storage.buckets.list / storage.objects.get / storage.objects.list
```

**一条 `artifactregistry.*` 都没有。** 它只能读 GCS（那是给源码/pickle 部署用的）。
所以走 `containerSpec` 必须补这一条。**把邮箱放进变量再用**，别把它写在
`--member=` 后面（理由见下）：

```bash
SA=service-596821501265@gcp-sa-aiplatform-re.iam.gserviceaccount.com

# 幂等，确保服务代理已创建。它的输出只会显示 Vertex AI Service Agent
# （不带 -re 的那个），但文档明说 Reasoning Engine Service Agent 也一起建了。
gcloud beta services identity create --service=aiplatform.googleapis.com \
    --project=study-project-496907

# 仓库级，最小权限
gcloud artifacts repositories add-iam-policy-binding cctv-audit \
    --location=us-central1 --project=study-project-496907 \
    --member="serviceAccount:$SA" --role=roles/artifactregistry.reader
```

### 踩过的坑：`INVALID_ARGUMENT: Invalid service account`

```
ERROR: Policy modification failed. For a binding with condition, run
"gcloud alpha iam policies lint-condition" to identify issues in condition.
ERROR: INVALID_ARGUMENT: Invalid service account
(service-596821501265@gcp-sa-aiplatform-re.iam.gs
  erviceaccount.com).
```

**邮箱没错，是复制粘贴时被折行了。** 看清楚 `gs` 和 `erviceaccount.com`
中间断开了——双引号里的换行是字面量，gcloud 收到的确实是个非法字符串。
那句「run lint-condition」的提示是误导，跟 condition 一点关系都没有。

这个邮箱 66 个字符，加上 `--member="serviceAccount:` 前缀就超过 80 列，
在终端里粘贴极容易断。**所以用变量。**

**授权这一步客户环境也要做**——不是我们这个项目的偶发问题，是 BYOC 的固定前置步骤，
真实交付时要写进部署清单。

### 另一条路：`sourceCodeSpec`，**试过，也没成**

discovery document 里还有 `ReasoningEngineSpec.sourceCodeSpec`：

```
inlineSource.sourceArchive   源码 .tar.gz，base64 直接塞进请求体
imageSpec {}                 «归档里有 Dockerfile，构建它»
pythonSpec                   或者不给 Dockerfile，按 entrypoint 装 Python 包
```

想法是让平台用它自己的 Cloud Build 构建，全程不碰我们的 registry，
这样就不需要上面那条授权。`create_engine.py create-src` 走的就是这条。

**实测第 4 行：跑了 9 分钟才失败**（前几次都是秒级），说明它确实在构建，
但最后还是 code 13，而且**没有任何容器日志、我们项目的 Cloud Build 列表里也没有这次构建**
（构建在 Google 侧跑）。所以失败在构建阶段还是启动阶段，目前**分不出来**。

没有继续深挖，因为第 3 行已经把 `containerSpec` 的原因钉死了，
补一条授权就能验证；而这条路连错误信息都拿不到。
**授权之后如果 `containerSpec` 通了，这条就不用管了**；
如果客户环境不允许给服务代理授权，再回来啃它。

两个入口都留着：`create` 是 `containerSpec`，`create-src` 是 `sourceCodeSpec`。

---

## GE 怎么注册外部 agent（计划里标「没找到文档」的那条，现在有答案了）

散文文档没有，v1alpha 的 discovery document 里有：

```bash
curl -s "https://discoveryengine.googleapis.com/\$discovery/rest?version=v1alpha"
```

```
projects.locations.collections.engines.assistants.agents   create / get / list / patch / delete
```

`Agent` 这个资源上有 `adkAgentDefinition.provisionedReasoningEngine.reasoningEngine`
—— **就是一个指向 `projects/.../reasoningEngines/{id}` 的资源名**，没有别的打包或发布步骤。
另外三种挂法是 `a2aAgentDefinition`（贴一张 JSON agent card）、
`dialogflowAgentDefinition`、`managedAgentDefinition`（Google 自带的，比如 Deep Research）。

顺带确认了两件影响计划的事：

- **Prompt chips 就是 `starterPrompts`，每一项只有一个字段 `text`。**
  **没有隐藏参数的位置。** 所以计划里「chip 文案给人看、chip 携带版本号给机器看」
  这个分法**做不到**——`sop_id` 只能写在用户看得见的文案里
  （比如「按门店标准作业规范 v3 稽核」，后面 GE 的 Instructions 负责把 `v3` 提出来），
  或者干脆一个 SOP 版本配一个 agent。这条要回改计划第 3 点。
- `authorizationConfig` 分两种：`agentAuthorization` 走请求头，
  `toolAuthorizations` 走请求体。拿到 CCTV 控制台账号以后是这里接。
- **`agents.create` 不校验 `reasoningEngine` 指向的资源存不存在。** 拿一个
  全是 0 的假资源名去建，照样返回 `"state": "ENABLED"`。
  意思是**资源名打错了要到运行时才炸**，配置里得自己核对。（那个测试 agent 已删掉。）

建应用和注册 agent 都在 `create_ge_app.py`：

```bash
python deploy/phase0/create_ge_app.py create-app          # 建 GE 应用
python deploy/phase0/create_ge_app.py create-agent <reasoningEngine 资源名>
python deploy/phase0/create_ge_app.py list-agents
```

注意 discoveryengine **必须带 `X-Goog-User-Project` 头**，否则 403
（「requires a quota project, which is not set by default」）；
而且**只有 `global`**，填 `us` 直接 400。

---

## GE 到底怎么调我们的容器（实测抓包，**文档一个字都没有**）

这一段是 Phase 0 最值钱的产出，也是最反直觉的一条。

### 它只会调一个方法

不管 `classMethods` 里声明了多少个方法，**GE 从头到尾只发这一个请求**：

```
POST /api/stream_reasoning_engine
{"class_method": "streaming_agent_run_with_events",
 "input": {"request_json": "{\"message\":{\"role\":\"user\",\"parts\":[{\"text\":\"稽核 XX 店\"}]},
                             \"session_id\":\"...\",\"user_id\":\"you@example.com\"}"}}
```

三件事要注意：

1. **`preflight` / `start_audit` / `get_status` 在 GE 眼里根本不存在。**
   它不会把它们当三个工具来挑。每个方法上写的 `description` 和 `parameters`
   在这条路上**完全没用**——那是给 `:query` 直调准备的。
   → **计划第 1 点那个「三个方法」的拆法在 GE 这条路上不成立。**
   分流只有两个地方能做：**在容器里自己分**（看这轮说了什么），
   或者**一个操作配一个 GE agent**。这是 Phase 1/2 的硬约束，不是风格问题。
2. `request_json` 是 **JSON 里套一个 JSON 字符串**，双层编码。
3. `api_mode` 必须写 **`async_stream`**，不是 `stream`。
   这个值抄自 ADK 自己的部署模板 `google/adk/cli/cli_deploy.py`——
   一个真 ADK agent 注册的就是这一串。

### 回什么它才认

每一行必须是这个信封，**光发事件本身会被静静丢掉**
（错误信息只有一句 `stream closed cleanly without producing any events`）：

```json
{"events": [ <ADK Event>, ... ], "session_id": "..."}
```

Event 本身是 ADK 的 `Event.model_dump_json(exclude_none=True)`——snake_case、
不带 null，至少要有 `content{parts,role}` / `invocation_id` / `author` /
`actions{...}` / `id` / `timestamp`。分帧是 `json.dumps(chunk) + "\n"`，
`media_type` 是 `application/json`（不是 ndjson）。

### 怎么找出来的：**别猜，去读实现**

这里连着猜错了三轮——以为 chunk 就是 event、以为要双层编码、A/B 试了两种编码。
两种都失败说明**编码根本不是那个轴**。真正解决问题的是去读参考实现：

```bash
# 平台跑 ADK agent 时用的模板，信封和方法名都在里面
unzip -p ~/.cache/.../google_cloud_aiplatform-2.1.0-*.whl \
    vertexai/agent_engines/templates/adk.py > /tmp/adk_template.py
# 分帧和 media_type 在这里
.venv/lib/python3.10/site-packages/google/adk/cli/fast_api.py
```

**discovery document 比散文文档权威，而 SDK 源码比 discovery document 还权威。**
遇到「文档没写」的时候，先想想哪个包里有参考实现，再动手试。

### 两条 GE 调用路径，行为不一样

| 路径 | 结果 |
| :--- | :--- |
| v1 `.../agents/{id}/a2a/v1/message:send` | ✅ **直达我们的 agent**，每次都是 |
| v1alpha `assistants:streamAssist` + `agentsSpec.agentSpecs[].agentId` | ❌ **不路由到我们**，回话的是 GE 自带的助手 |

`streamAssist` 那条试了两次，第二次还专门把提问写得贴着 agent 的描述，
照样是自带助手回的「我访问不了门店监控数据」。
所以 **`agentsSpec` 不能用来指定路由**，测我们自己的 agent 要走 A2A。
（`assist.py` 走 streamAssist，`a2a.py` 走 A2A——留着两个是因为这个差别本身就是结论。）

---

## 本地先跑一遍

部到云上之前先在本机确认探针本身没问题：

```bash
./.venv/bin/python -m uvicorn probe:app \
    --host 127.0.0.1 --port 8099 \
    --app-dir deploy/phase0 --timeout-keep-alive 3600 &

./deploy/phase0/measure.sh http://127.0.0.1:8099 5 10
```

停的时候**别用 `pkill -f`**——模式会匹配到发起命令的 shell 自己，
把自己的终端杀掉（这个坑已经踩过两次了）。按端口找 PID：

```bash
kill "$(ss -lptn 'sport = :8099' | grep -oP 'pid=\K[0-9]+' | head -1)"
```

---

## 部到 Agent Runtime

> 这几步会真的创建云资源、开始计费。**先看一眼再执行。**

项目是 `study-project-496907`，location 按现在的口径统一用 `global`。

```bash
PROJECT=study-project-496907
REGION=us-central1          # Artifact Registry 要一个真实 region，不能用 global
REPO=cctv-audit
IMAGE="$REGION-docker.pkg.dev/$PROJECT/$REPO/phase0-probe:v1"

# 1. 建仓库（只需一次）
gcloud artifacts repositories create "$REPO" \
    --repository-format=docker --location="$REGION" \
    --description="CCTV audit agent images"

# 2. 构建并推镜像
gcloud builds submit deploy/phase0 --tag "$IMAGE" --project "$PROJECT"

# 3. 授权（见上面「部署卡在哪」，不做这一步必然失败）
#    邮箱一定要走变量，直接粘贴会折行，报出来的错还会把你带偏。
SA=service-596821501265@gcp-sa-aiplatform-re.iam.gserviceaccount.com
gcloud beta services identity create --service=aiplatform.googleapis.com --project="$PROJECT"
gcloud artifacts repositories add-iam-policy-binding "$REPO" \
    --location="$REGION" --project="$PROJECT" \
    --member="serviceAccount:$SA" --role=roles/artifactregistry.reader

# 4. 建 reasoningEngine
ENGINE_LOCATION=us-central1 python deploy/phase0/create_engine.py create
python deploy/phase0/create_engine.py poll <operation-name>
```

`create_engine.py` 直接打 REST，没用 `google-cloud-aiplatform`
（venv 里的版本不够新，而且每个字段都是从 discovery document 抄的，可以对着源头核）。
两个环境变量：`ENGINE_LOCATION`（默认 `global`）和 `ENGINE_SERVICE_ACCOUNT`
（不设就用默认服务代理；**设了以后错误信息会详细得多**，排障时值得设上）。

`classMethods` 七个方法都声明了，`api_mode` 按 contract：`""` 一问一答，
`"stream"` 流式，`"async_stream"` 是 GE 那个入口专用。

> ~~每个都带 `description` 和 `parameters`——schema 说这是「OpenAPI
> specification format」，**GE 是靠这两样决定调哪个方法、传什么参数的**，
> 不写就等于对 GE 隐身。~~
>
> **这段是错的，实测推翻了。** GE 根本不挑方法，只调
> `streaming_agent_run_with_events` 一个。`description`/`parameters` 只对
> `:query` 直调和 playground 有意义。详见上面「GE 到底怎么调我们的容器」。

资源参数按计划里那张表——探针不吃资源，`cpu=1` / `memory=1Gi` 就够，
但 **`min_instances=1`**：设成 0 的话冷启动会污染超时测量。

部好以后：

```bash
./deploy/phase0/measure.sh <reasoningEngine 端点> 60 300 900 1800
```

---

## 测量结果（跑完填这里）

**这张表是 Phase 0 的唯一交付物。** 后面每一步都建立在它上面。

### 问题 1：能挂多久

#### Agent Runtime 那一层（`:query` 直调，引擎 A）

| 探针 | 客户端看到 | 服务端记录 |
| :--- | :--- | :--- |
| hang 60s | ✅ 61s 返回 | 干完了 |
| hang 300s | ✅ 301s 返回 | 干完了 |
| hang 900s | ❌ **896s 被掐**，HTTP 400 | **干完了** |
| hang 1800s | ❌ **896s 被掐**，HTTP 400 | **干完了** |
| stream 1800s（每 10s 一个心跳） | ❌ **602s 就断了**，只收到 61/180 个心跳 | |
| `asyncQuery` | ⚠️ **发起成功，但什么都没发生**（见下） | 容器一个请求都没收到 |

**天花板是 896 秒（约 15 分钟），不是推断的 1 小时。**
900、1800 和另跑一次的 1000 都断在同一个数上——三次同一个数，
说明这是个固定的上限，不是网络抖动。`keepAliveProbe.maxSeconds` 3600 和 Cloud Run 的 60 分钟
**这两个旁证都指错了方向**，实测比推断早了四倍。

**但服务端每次都把活干完了，实例也没换。** 这一条比上限本身更重要：
连接断了不代表任务死了，所以**「发起 → 断开 → 事后取结果」是成立的**。

被掐的时候客户端拿到的是一句**没有任何信息量**的话——注意它连
「超时」两个字都没有，`Error Details:` 后面是空的：

```json
{"code": 400, "status": "FAILED_PRECONDITION",
 "message": "Reasoning Engine Execution failed.\nPlease refer to our
             documentation (...) for checking logs and other troubleshooting
             tips.\nError Details: "}
```

**别被它带偏去查容器日志**（容器那边什么错都没有，活干得好好的）。
GE 那条路反而说了实话，明写 `DEADLINE_EXCEEDED`——见下。
排障时如果看到这句空洞的 `Reasoning Engine Execution failed`
又刚好等了 15 分钟左右，**先怀疑超时，别怀疑自己的代码**。

#### 流式**没有**买到更多时间，反而更少

这一条推翻了本文件上面「超时机制通常只惩罚沉默」那个假设，**写在这里提醒别再犯**。

一个每 10 秒吐一次字节、一直没停过的流，**602 秒就断了**——
比那个从头安静到尾的 `hang` 还早了将近五分钟。所以：

- **「一边跑一边往 GE 推进度」撑不满一场稽核。** 计划第 5 点那张表里
  「都能挂住 ≥30 分钟 → 流式推进度」这条路**直接排除**，10 分钟就到顶。
- 心跳保活是**没用的**，这里掐的不是 idle timeout。
- 真正的上限反而是 unary 的 896s。要在一个请求里做完事，那 15 分钟是全部预算。
- 而且流式这条路**连活都保不住**——容器侧 900s 被 cancel，见下面那一节。

→ 三条路里只剩**轮询**和**跑完主动通知**。而 896s 这个数对一场几十分钟的稽核
来说不够，所以 **`start_audit` 必须立即返回、后台跑**，没有别的选择。

#### GE 那一层（A2A `message:send`，引擎 B）

| 探针 | 结果 |
| :--- | :--- |
| hang 60s | ✅ 60.0 秒后正常返回 |
| hang 3600s | ❌ **602.8s 被掐** |

```json
{"code": 400, "status": "FAILED_PRECONDITION",
 "message": "Agent failed with error: Reasoning Engine Execution Service
             stream failed with status code DEADLINE_EXCEEDED: ",
 "reason": "REMOTE_AGENT_FAILURE", "domain": "discoveryengine.googleapis.com"}}
```

**602.8s 和上面那个流式的 602s 是同一个数**，而且报错里明说是
`Reasoning Engine Execution Service stream ... DEADLINE_EXCEEDED`。
所以这不是「GE 的超时」——**GE 只是把 Agent Runtime 流式那一层的 deadline 转述了一遍**。

于是原本以为的两层其实是一层，一共只有两个数：

| 路径 | 上限 |
| :--- | ---: |
| `:query`（一问一答） | **896s** |
| `:streamQuery` / GE（GE 永远走流式） | **602s** |

**GE 那条路只有 10 分钟。** 而且我们在 GE 后面没得选——
它只会调 `streaming_agent_run_with_events`，也就是只会走流式那条路。

#### ⚠️ 流式这条路，**活会被杀掉**（和 unary 不一样）

这一条一开始判错了，记在这里因为它直接决定 Phase 2 的代码怎么写。

那次 3600 秒的 hang，GE 在 602s 就报错走了。当时去查 `probe_log`（容器时间 1275s）
看到任务还在跑、没有 cancel，就以为「和 unary 一样，断的只是连接」。**错了——
查早了 81 秒。**

```
 456.6  ge_turn             3600     ← GE 转发进来
1059    （GE 在这里 DEADLINE_EXCEEDED 退出，容器毫无反应）
1356.6  ge_hang_cancelled    900     ← 456.6 + 900，一秒不差
```

引擎 A 上的流式测试是同一个数：`stream_start` 2427.9 → `stream_cancelled` 3327.9，
**整整 900.0 秒**。两个引擎、两条不同的流，都在 900s 被掐。

所以两条路的行为**根本不同**：

| 路径 | 客户端断在 | 容器里的活 |
| :--- | ---: | :--- |
| `:query` 一问一答 | 896s | ✅ **跑到底**（1800s 和 1000s 两次都记到了 `hang_completed`） |
| `:streamQuery` / GE | 602s | ❌ **900s 被 cancel** |

**对 Phase 2 的直接要求：稽核绝对不能跑在 GE 那个请求的协程里。**
`streaming_agent_run_with_events` 收到「开始稽核」之后，
必须把活**甩进一个脱离本次请求的后台任务**（`asyncio.create_task` 存到模块级、
或者干脆另起线程），然后立刻返回 job_id。
写成 `await run_audit()` 的话，900 秒一到整场稽核就没了，**而且 GE 那边 602 秒
就已经报错了，根本看不到是怎么死的**。

「立即返回 job_id + 后台跑 + 轮询取结果」因此不是为了体验好，**是唯一能活下来的写法。**

#### `asyncQuery`：**别把它当后路，它现在是通不了的**

计划里一度把 `asyncQuery` 当成「超过一小时的长任务的官方解法」。实测跑了一次，
结论是**现在指望不上**，原因还没查清。

请求体和 `:query` **完全不一样**——没有 `classMethod` 和 `input`，
只有两个 GCS 路径，要调什么方法得写进文件里：

```bash
printf '{"class_method":"hang","input":{"seconds":30}}' > aq-input.json
gcloud storage cp aq-input.json gs://<bucket>/phase0/aq-input.json

curl -X POST ".../reasoningEngines/<id>:asyncQuery" \
  -d '{"inputGcsUri":"gs://<bucket>/phase0/aq-input.json",
       "outputGcsUri":"gs://<bucket>/phase0/aq-output.json"}'
# → {"name": ".../operations/1476033221429821440"}
```

然后**就没有然后了**。十分钟后：

- `operations.get` 回的还是**光秃秃一个 name**——没有 `done`、没有 `metadata`、
  没有 `error`，什么都没有
- `outputGcsUri` 那个文件**没被创建**
- **容器侧一个请求都没收到**（`probe_log` 里只有我自己那几次 `probe_log`）

所以它**卡在平台里**，压根没走到我们这一层。是输入文件格式不对
（schema 只说「BYOC 的话内容取决于你的应用」，等于没说），
还是 BYOC 上根本没接，**分不出来**——错误信息一个字都没有。

**影响：**「长任务走 `asyncQuery`」这条后路暂时不能算数。
好在也不需要了——上面已经证明「后台任务 + 轮询」这条路是通的。
真要用它，得先跟 Google 问清楚 BYOC 的输入文件到底该长什么样。

#### 回头看：**当初那个「约 1 小时」的推断，错得挺离谱**

事先从 schema 里推出来的是「约 1 小时」——`keepAliveProbe.maxSeconds` 上限 3600s，
`resourceLimits` 链到 Cloud Run 而 Cloud Run 请求超时上限 60 分钟。两个旁证互相印证，
当时看着挺硬。

实测：**896s 和 602s**，比推断短了四到六倍。

那两个数不是假的，只是**量的根本不是这件事**——keepAlive 探针的周期上限
和一次请求能挂多久没有关系；Cloud Run 的 60 分钟是 Cloud Run 的上限，
Agent Runtime 在它上面又叠了自己更紧的一层。

**教训：两个旁证互相印证，不等于它们指的是同一个东西。**
凡是能实测的数，就别拿 schema 推。这也是 Phase 0 存在的理由。

结论 → 第 4 步接哪种：

- [x] ~~都挂得住（≥30 分钟）→ 流式推进度~~ **排除**：流 602s 就断，比 unary 还短
- [x] **只能挂几分钟 → 轮询 `get_status`**。unary 上限 896s，一场稽核放不进去
- [ ] GE 不支持长等待 → 跑完主动通知（邮件 / Doc / Chat），和 Workspace 那步合流
      （GE 侧的数还在测，见下面那张表）

### 问题 2：GE 能不能挂上去 — ✅ **能，端到端跑通了**

注册方式就是上面那个 `agents.create` + `provisionedReasoningEngine`。
**光注册成功不算数**（它连资源名存不存在都不校验），得有一次真的往返：

```
$ python deploy/phase0/a2a.py "say hello"
  探针在线。实例 398fc6b3-pid1，已运行 155.4 秒，收到「say hello」。
  [agent] CCTV 稽核探针
  [state] SUCCEEDED
```

回话里带着**容器自己的实例 id**，`diagnosticInfo.plannerSteps` 里也能看到
GE 的 planner 走进了我们的容器。这才算证据。

### 问题 3：能不能中途等确认 — ✅ **能**

两轮，中间靠 `A2aV1Message.contextId` 串起来（`a2a.py --context`）：

```
turn 1  稽核 XX 店 昨天 14:00-15:00
     →  找到视频了：Plan B 录屏，14:00-15:00 时间段够。确认开始稽核吗？
        contextId: .../engines/cctv-audit/sessions/15301453483600566158
turn 2  确认   （带上同一个 contextId）
     →  收到确认，等了 7.5 秒。上一轮问的是「稽核 XX 店 昨天 14:00-15:00」。
        同一个实例 398fc6b3-pid1，session_id 传回来了。
```

**两件事同时成立**：GE 愿意把一个问题抛给用户再拿着答复回来；
`session_id` 会原样传回容器，所以它是个可靠的状态 key。

⚠️ **但这次第二轮落回了同一个实例，那是运气不是保证**——探针 `min_instances=1`，
本来就只有一个。上面那句话存在探针的内存字典里，真实环境 `max_instances>1`
时第二轮完全可能落到别的容器上。**结论不变：状态必须外置**（计划里 Firestore 那条），
`session_id` 正好就是主键。

另外 A2A 的任务态里有 `TASK_STATE_INPUT_REQUIRED`——形状上是给「等用户输入」
准备的。这次没用上（我们靠 contextId 就够了），但真要做成正经的任务流可以往那看。

### 问题 4：出网 — ✅ **通**（在部署好的引擎上实测，不是本机）

| 目标 | reachable | status |
| :--- | :--- | :--- |
| `google.com/generate_204` | ✅ | 204 |
| `bilibili.com` | ✅ | 200，121430 字节 |

`any_reachable: true`。**不用配 Private Service Connect，也不用 Agent Gateway。**

本机基线（已测）：两个都通，204 / 200，bilibili 回 115 KB。
注意 **`reachable` 和 `status` 是两回事**：不带 UA 时 bilibili 回 412 拦截页，
带上正常 UA 就是 200。**412 也算通**——包出去回来了。
把 412 当成「不能出网」会白白去配一遍 Private Service Connect。

---

## 跑完之后

**Phase 0 收工了。** 四个问题都有答案和证据（`results/` 里是原始日志），
计划文件里「我没能验证的事」那一节也已经同步改过。**可以开始 Phase 1。**

这个目录留着——Phase 2 换真镜像时 `measure.sh` 还能拿来对比，
`a2a.py` 是唯一能直接打到我们自己 GE agent 的工具，后面每次改容器都要用。

### 还欠两笔（都不挡 Phase 1）

- **`asyncQuery` 为什么没动静**。要么问 Google BYOC 的输入文件格式，要么放弃。
  现在的方案不依赖它。
- **900s cancel 是不是可配**。`deploymentSpec` 里没找到对应字段；
  真有的话后台任务那套还能简化一点，但别指望。

### 两个引擎还在跑，**要么关掉要么记着它在花钱**

`min_instances=1`，也就是**没人调也一直有个容器开着**：

```bash
python deploy/phase0/create_engine.py list          # ENGINE_LOCATION=us-central1
python deploy/phase0/create_engine.py delete <资源名>
```

引擎 B（`2184621302495576064`）连着 GE agent，**删了 GE 那边就调不通了**，
要留到 Phase 4 对照的话就留着。引擎 A（`645797604818419712`）只是测超时用的，
测完就没用了。
