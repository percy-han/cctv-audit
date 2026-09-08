# 两个分支的对比 — 2026-09-08

`main` 和 `refactor/video-pipeline` 在 `29cbb38 initial project` 之后各自往前走了，
而且**两边都做了上云**。这份文件把两套方案摊开摆在一起，供讨论用。

- `main` = `0b9664a`，关键提交 `5bc2cd2 Add GE and Cloud Run deployment architecture`（2026-09-03）
- 本分支 = `c2fcecf`，从 `29cbb38` 分出来，做了两轮（视频流水线 + 上云）

**目前不合并。** 这份文件不是在选谁赢，是把「各自解决了什么、各自欠着什么」写清楚，
让合并的时候能按条目挑，而不是按分支挑。

---

## 一句话

**两条线解决的是不同层的问题。**
`main` 解决的是「怎么把这东西部出去、让人能点」；
本分支解决的是「判定结论能不能拿去跟客户对峙」。

**`main` 的部署形态在某几个点上比本分支干净**，这个要先说。

---

## 架构：分工切在不同的地方

```
main:   GE / 调用方 ──▶ Agent Engine（ADK 托管打包，不是自定义容器）
                            │ OIDC ID token
                            ▼
                       Cloud Run 一个服务
                       ├─ 浏览器（Playwright + Computer Use 循环）
                       ├─ 稽核逻辑
                       ├─ 大屏 + WebSocket
                       └─ REST API（/api/v1/audit/*，配 openapi.yaml）
                       状态：进程内存里的 dict

本分支: GE ──▶ Agent Runtime 自定义容器
                  ├─ 浏览器（确定性 Playwright + Computer Use 兜底）
                  ├─ ffmpeg 采集 + 切窗
                  ├─ 稽核（Gemini 原生视频理解）
                  └─ 推预览帧 ──▶ Cloud Run 大屏（纯观看，无 LLM）
                  状态：Firestore（按 user_id 分子集合）+ GCS
```

**最大的结构差异：浏览器放在哪。**
`main` 放 Cloud Run，本分支放 Agent Runtime 容器。
下面几乎所有的优劣都是从这一个选择衍生出来的。

---

## `main` 这几天做了什么（它不是原始版本了）

`5bc2cd2` 里有几处实打实的改进：

| 改进 | 位置 |
| :--- | :--- |
| 结构化工具声明，`severity` 是 `RED_LINE / NORMAL / NONE` 枚举 | `agent_loop.py:111` |
| **只保留最近 2 张截图**，老的从上下文剪掉（控上下文和成本） | `agent_loop.py:729` |
| CDP screencast 推大屏，不再一帧帧截图推 | `agent_loop.py:643-662` |
| 一整套 Cloud Run 部署 + OpenAPI + 部署文档 | `deploy_cloud_run.sh` / `openapi.yaml` / `DEPLOYMENT_GUIDE.md` |
| 任务甩后台、立刻返回 `task_id` 和大屏链接 | `monitor_server.py:516` |

**最后那条两边独立走到了一起。** 本分支是被实测逼出来的
（GE 那条路 602 秒断，流式请求里的活 900.0 秒被 cancel，见 `deploy/phase0/README.md`），
`main` 是直接就那么写的。**同一个结论两条路各自到达，说明这个方向是对的。**

---

## `main` 比本分支强的三点

### 1. `--no-cpu-throttling`：它从一开始就没有「后台任务挨饿」这个 bug

`deploy_cloud_run.sh:78`。CPU 常驻分配，后台任务照样有 CPU。

本分支在 Agent Runtime 上被这件事坑了整整一轮：**没有请求在飞的时候拿不到 CPU，
runqueue wait 79–85%，大屏掉到 0.8 fps 幻灯片**，最后只能用
「把 `/is_busy` 的响应挂住不返回 + `keepAliveProbe` 定时来敲」
构造出「一直有请求在飞」来绕。

**Cloud Run 有一个开关直接解决，Agent Runtime 没有这个开关。**
这是平台选择带来的差别，不是谁代码写得好。

### 2. 一个可部署单元，不存在版本对不齐

本分支是引擎 + 大屏两个服务，必须同 tag 部署。
**这个坑真的踩了**：大屏在 Cloud Run 上停在 `v14`，而文档里写的是 `v17`，
中间隔了好几版才发现。`main` 只有一个服务，结构上不可能出现这种事。

### 3. 链路短

`gcloud run deploy --source .` 一条命令。
本分支是三步（build → 引擎 → 大屏），还得记住
「改了 spec 要用 `update-spec` 不是 `update-image`」——
用错了的话镜像换了、配置没换，症状和没部一模一样。

另外 `--session-affinity`（WebSocket 粘连）也考虑到了，这条本分支的大屏同样设了。

---

## `main` 上还欠着的（按严重度）

### 1. 模型从头到尾只看截图，没有视频

`agent_loop.py:694` 是唯一喂给模型的东西：

```python
Part.from_bytes(data=screenshot, mime_type="image/png")
```

CDP screencast（`:643`）**只推大屏，不进分析**。全仓库没有 ffmpeg。

所以本分支第一轮要解决的那个问题在 `main` 上原样存在：
**SOP 五条里四条是「没做某事」（没洗手、没冲洗、没加盖、看手机），
而「没做」这种判断在单帧上不可证伪。**
按 25fps 算，每 10 秒看一帧的时序覆盖率是 **0.27%**。

这是两个分支之间**唯一一条影响交付结论可信度**的差异，其余都是工程形态。

### 2. 关键词正则还在，而且能凭空造出红线违规

```python
# agent_loop.py:163  extract_and_record_segments_from_text(text)
# agent_loop.py:184
severity = "RED_LINE" if is_viol and any(
    rw in desc_raw for rw in ["口罩", "帽子", "手套", "红线"]
) else "NORMAL"
```

它对模型的**思考文本**跑匹配。模型只要写一句「检查是否有口罩违规」，
就会被记成一条红线违规。

注意上面提到的结构化枚举（`:111`）**没有取代它，是并存的**——
两条记录来源同时存在，一条来自模型的结构化输出，一条来自对文本的猜测。

### 3. 状态全在进程内存里

```python
# monitor_server.py:386
active_tasks: Dict[str, Dict[str, Any]] = {}
```

Cloud Run 实例一换（部署、缩容、崩溃），所有任务凭空消失，
`/api/v1/audit/status/{task_id}` 直接 404，客户看到的是「任务不存在」。

### 4. 第二个人发起稽核，会把第一个人的掐掉

```python
# monitor_server.py:427-434
current_background_task.cancel()   # 记成 "被新发起的抽检任务取代"
```

配合 `MAX_INSTANCES=1`（`deploy_cloud_run.sh:20`），
**全公司同时只能跑一单**。第二个人一点，第一个人的稽核就没了，
而第一个人只会看到自己的任务变成 CANCELLED。

### 5. 没有用户隔离

`monitor_server.py:603` 的 `list_audit_tasks()` 把所有人的任务原样返回，
`get_audit_status()` 也不校验调用者是谁。
任何有 `run.invoker` 的人能看到所有人的稽核内容和违规详情。

本分支是 Firestore 按 `{root}/{user_id}/jobs/{job_id}` 分子集合，
**串号在结构上就不可能**，不依赖「记得加过滤」。

### 6. 三处小的

- `deploy_cloud_run.sh` 头部注释写 `--concurrency 3 prevents Chromium RAM exhaustion`，
  实际默认是 `CONCURRENCY=80`（`:21`）。注释和代码对不上。
- `MAX_AUDIT_TURNS` 默认 `100000`（`agent_loop.py:597`），等于没有上限。
- 默认项目是另一个项目（`cs-poc-hzdu6g9fvdacmw21rd6jq89`）、
  默认 URL 写死了别的项目号，在本项目跑要带参数覆盖。

---

## 本分支的代价（公平起见）

- **多一个服务、多一层部署**，两边版本要对齐——已经栽过一次。
- **Agent Runtime 上没有 `--no-cpu-throttling`**，只能靠挂住探针换。
  连带一个必须一直记住的约束：**挂住的探针占一个并发槽，
  所以 `containerConcurrency` 必须 ≥ 2**，否则 GE 的正常轮次会吃 429。
- **复杂度高得多**：20732 行 vs 4994 行。多出来的主要是测试（6012 行）
  和状态外置那一层。
- **云上的容器还进不去 bilibili**（登录态不在镜像里，这是刻意的）。
  `main` 同样进不去，只是它没做这项验证，所以还没暴露。

---

## 真要合的时候，建议按条目挑

| 取谁的 | 什么 | 为什么 |
| :--- | :--- | :--- |
| 本分支 | 视频流水线 + 结构化判定 | 这是产品价值本身，不是工程偏好 |
| 本分支 | Firestore 状态外置 + 按人隔离 | 内存态在 Cloud Run 上必然丢 |
| 本分支 | 380 个单测 | 不碰浏览器/网络/Vertex，跑一次 16 秒 |
| **`main`** | **Cloud Run + `--no-cpu-throttling` 这套部署形态** | 浏览器要是能搬过去，`keepAliveProbe` 那套 hack 可以整个删掉 |
| `main` | `openapi.yaml` 这种对外契约的写法 | 交付给客户时用得上 |
| 都别要 | `extract_and_record_segments_from_text` | 能凭空造红线违规 |
| 都别要 | `MAX_AUDIT_TURNS=100000` | 等于没有上限 |
| 都别要 | 内存里的 `active_tasks` | 见上 |

**最值得先聊的一件事**：浏览器到底该放哪。

本分支当初选 Agent Runtime，是因为 GE 只认 `reasoningEngine` 资源；
但 `main` 的做法说明「Agent Engine 当薄壳、真活在 Cloud Run」这条路也存在。
**如果它能在 GE 里真的跑通，工程上确实更省事**——
一个服务、一个开关解决 CPU、没有版本对齐问题。

---

## 我没验证的，别当结论

- **`main` 那条链路能不能在 GE 里真的跑通，我只读了代码，没跑过。**
  尤其是 Agent Engine（ADK 托管打包那种）转发到 Cloud Run 这一跳。
  本分支的 602s / 900s 那些数是在**自定义容器**上量的，
  不保证在 ADK 托管形态上是同一组数。
- `main` 的 `--no-cpu-throttling` 能解决挨饿，这是按 Cloud Run 的语义推的，
  **没有在 `main` 那套部署上实测过 runqueue wait**。
- 两边的 token 数没有在同一段素材上对比过，
  「截图 vs 视频谁更贵」现在还是空白。

---

## 另外两件需要人决定的

我不单方面动别人分支上的东西，所以只记在这里：

- `main` 上提交了 `monitor_server.log`（20487 行）和
  `computer_use_agent/checkpoints/audit_records.jsonl`（28 条稽核记录）。
  本分支的 `.gitignore` 把 `checkpoints/` 整个挡掉，理由是**稽核记录涉及可识别的人**。
  `main` 上没有证据帧 JPG，只有记录和日志。
- 仓库现在是 private，所以上面这条不是外泄，
  是**要不要留在 git 历史里**的问题。
