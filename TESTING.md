# 怎么测这套东西

按「最快见效」排，从上往下越来越慢越来越真。
所有命令都在仓库根目录 `~/Code/cctv-audit` 下敲。

> **这台开发 VM 的 gcloud CLI 登录是坏的**（非交互没法重认证），
> 但 ADC 正常。所以下面所有脚本都走 ADC + REST，不依赖 gcloud CLI。
> 只有 `gcloud run services proxy` 那条要在**你自己的笔记本**上敲。

---

## 1. 单元测试（10 秒）

```bash
.venv/bin/python -m pytest -q
```

最近一次：**380 passed**。全是纯逻辑，不开浏览器、不连云，改完代码先跑这个。

## 2. 收工自查（1 分钟）

```bash
./check.sh          # 快
./check.sh --deep   # 还会从 GitHub 重新 clone 一份比对，证明代码离开这台机器也活得了
```

## 3. 本地跑一整场稽核（2–3 分钟）★ 最有用

**这是判断「是代码的问题还是云上的问题」的唯一办法。**

脚本在 **`/tmp/bili_v18.py`**（上一版 `/tmp/bili_v12.py` 还在，少了下面那两个数）。
它比老的 `/tmp/bili_repro.py` 多做一件要紧的事：
**自己当一个观看端连上大屏的 WebSocket**。
预览帧是靠 `has_viewers` 门控的，没人看的那种跑法**根本没走预览这条路**——
而这条路正是被投诉「卡成幻灯片」的那条。

v18 又多做一件：**给每一帧算 md5**，然后报「不重样的画面有几张」和
「最长的一段一模一样的画面持续了几秒」。这是为了抓「弹登录框之后画面死了」那个 bug——
**它的帧率是满的、字节数是恒定的、所有指标都好看，只有画面是死的**，
光看 fps 一辈子发现不了。跑法：

```bash
PYTHONPATH=$PWD \
GCP_PROJECT=study-project-496907 GOOGLE_GENAI_USE_VERTEXAI=TRUE \
SOP_BUCKET=study-project-496907-cctv-audit DEFAULT_SOP_ID=chagee-store-v1 \
HUMAN_GATE_MODE=off PREVIEW_FPS=12 PREVIEW_WIDTH=1280 PREVIEW_HEIGHT=720 \
MONITOR_URL=http://127.0.0.1:8099 \
.venv/bin/python -u /tmp/bili_v18.py
```

`SOP_BUCKET` 不能省，preflight 会直接以「没有配置 SOP_BUCKET」退回。

（先在另一个窗口起本地大屏：`MONITOR_HOST=127.0.0.1 MONITOR_PORT=8099
.venv/bin/python -m computer_use_agent.monitor_server`。
**收工记得停掉**：`kill "$(ss -lptn 'sport = :8099' | grep -oP 'pid=\K[0-9]+' | head -1)"`，
**不要 `pkill -f`**——那个模式串在发起命令自己的命令行里，会把当前 shell 一起杀了。）

环境变量 `BILI_URL` / `START` / `SPAN` 换视频和时间段。
`AuditRequest` 的字段名是 `target` / `start_seconds` / `duration_seconds`
（**没有 `end_seconds`** 这个构造参数）。

最后会印一张表，两组数字各回答一个问题。
**2026-09-08 真 bilibili（单 `e40125`，v18，`BV1MisPzrE29`，01:00 起 5 分钟）：**

```
job e40125: done after 327s
  windows_analyzed       25
  windows_failed         0
  violations             4
  capture_mode           stream
  elapsed_seconds        288.7
  complete               True

  dashboard frames       3416
  over                   288s -> 11.9 fps
  frame size             64.3 KB avg, 7.5-120.4 KB      <- 会变，说明画面在动
  gap between frames     median 84ms, worst 127ms
  gaps over 1s           0            <- 「像视频」还是「像幻灯片」
  distinct pictures      3381 of 3416 <- 画面到底有没有在变
  longest identical run  1.1s (14 frames)
```

**后面三行是这一轮新加的，也是唯一能证伪「画面冻住」的三行。**
坏的时候长这样：帧率照样 11.9 fps、`gaps over 1s` 照样 0、
但**每帧恒定 84.3 KB、最长静止两分钟**。
`longest identical run` 超过 10 秒脚本会自己在后面标 `<-- THE BUG`。

同一场里另外三个该看的：

- 容器日志里要能看到 `Dismissed overlay .bili-mini-close-icon`
  （04:49:07，播放开始约 70 秒）。**这行只可能来自 `keep_clear`，
  而 Plan A 下只有页面看门狗会调它**——它出现就等于证明看门狗在 Plan A 上跑起来了。
- 弹窗前后的投屏帧率 24.9 → 24.6 fps，**没有掉到 0**。
- `Dashboard picture frozen` 和 `Chromium sent no screencast frame` 这两行
  **一次都不该出现**。出现了就是画面真的死过。

单窗口推理耗时：n=25，最快 14.9s，**中位 19.9s**，p90 22.5s，
每窗口约 10504 input token。
**有一个窗口 250.7 秒**——日志里没重试没报错，就是那次模型调用自己慢，
它把整场墙上时间拖长了三分半。这个还没查。

更早的对照：2026-09-04 老版本 68 秒跑完 2 处违规（演示素材，不是 bilibili）。

## 3.5 Plan C：稽核桶里的一个视频文件（`gs://`）

**这条路什么都不采集。** 对象地址直接交给 Vertex，字节不进容器，
**整段视频一次分析完，不切片**（原因见 `CHANGES.md` 十二·九：agentic 会
静默忽略时间偏移）。所以这里没有窗口数可数、没有采集耗时可比，
也**没有实时画面**。

**先决条件**：跑稽核的那个服务账号要在**放视频的那个桶**上有
`roles/storage.objectViewer`。放在 `study-project-496907-cctv-audit` 就不用再授——
引擎已经在从这个桶读 SOP、往里写证据帧。本地跑用的是你自己的 ADC，
云上用的是引擎的服务账号，**两个身份不一样，本地通不代表云上通**。

不用开浏览器，也不用模型，就能把 ffmpeg 那两件小事走完一遍：

```bash
PYTHONPATH=$PWD .venv/bin/python - <<'EOF'
import asyncio
from computer_use_agent.capture import gcs_video

URI = "gs://study-project-496907-cctv-audit/你的路径/视频.mp4"

async def main():
    src = await gcs_video.open_source(URI)          # 探一下：在不在、多长
    print(src.mode, src.duration_seconds, src.reason)
    jpg = await gcs_video.grab_frame(src, 300.0)    # 第 5 分钟抽一帧
    print("cover bytes:", len(jpg) if jpg else None)

asyncio.run(main())
EOF
```

`mode` 应该是 `file`，`duration_seconds` 要和 `ffprobe` 对得上。
**读失败的时候看它说了什么**——403 应该明写要加哪个角色，404 应该提醒对象名区分大小写。
只说 "Server returned 403 Forbidden" 就是没走到 `_explain`。

整场稽核走第 3 节那套，把 `BILI_URL` 换成 `gs://...` 即可：
`gs://` 走的是同一个 `AuditPipeline.run()`，只是 `_capture_session`
把它路由到了不开浏览器的那条分支，producer 换成 `WholeFileProducer`。

**要看的四件事**：

1. GE 的确认回复里是 `采集方式：直接读文件（GCS）` +
   `分析方式：agentic，整段视频一次看完，不切片` +
   `要稽核：整段视频，从头看到尾`。**说了时间段的话，回复里要明说这次用不上。**
2. 回「确认」之后的那句话里**没有「实时画面」这一行**。
3. 只有一条窗口记录，`time_range` 是 `00:00 - <总长>`。
4. 违规的证据截图有图——那是 `frame_at` 从桶里现切的，
   **这是整条路上唯一还会碰字节的地方**，坏了只会表现为「报告里没图」。

token 数值得记一笔：`AnalysisOutcome.input_tokens` 已经把
`tool_use_prompt_token_count` 加进去了，agentic 抓的帧全在那一项里，
只看 `prompt_token_count` 会少报八成。

> **2026-09-09 云上跑过一次，单号 `ad345a`**（镜像 v23，
> `gs://study-project-496907-cctv-audit/tmp/agentic-probe/store.mp4`，20 分钟）。
> `capture_mode=file`、`analysis_scope=whole_file`、`start_audit` 0.45 秒返回、
> 44.2 秒出报告、**1 个窗口 0 失败**、`time_range 00:00 - 20:00`、
> 覆盖 0→1200 且 `complete=True`、输入 9799 / 输出 1682 token。
> 封面帧确实从桶里切出来了（`audits/ad345a/cover.jpg`，10646 字节，真 JPEG），
> 所以「ffmpeg 按字节区间读 GCS」这半段是通的。
>
> **第 4 件事在 2026-09-09 补验了，单号 `292712`**（镜像 v24，
> `gs://study-project-496907-cctv-audit/poc-video/chagee-01.mp4`，5 分 02 秒，
> 客户给的真实门店录像）。判出 4 条红线违规，`frame_at` 按模型给的时刻
> 从桶里切了 4 张，全部落地：`audits/evidence/292712/w00000_*.jpg`，
> 95 763 – 101 629 字节。**逐字节存在不等于切对了地方**，所以其中一张
> （`CHK_BEHAVIOR_004_0293.jpg`，判定「员工看手机」）下载下来看过：
> 画面里就是那个人双手举着粉色壳的手机低头看。**切的位置对得上判定。**
> 之前那次（`ad345a`）验不了是因为素材是合成色块、一条违规都没有，
> `frame_at` 压根没被调到。
>
> **还有一个数不知道：Vertex 一次能吃多长的视频。** 限制是时长不是文件大小，
> 而这条路的前提就是「一次给一整段」。20 分钟已经证明可以，
> 再往上拿客户那个真正的大文件探一次。

## 4. 看大屏收到了什么（不用浏览器）

```bash
.venv/bin/python deploy/dashboard/watch.py <单号> --seconds 120
```

它连的是网页连的那同一个 `wss://.../ws?job=<单号>`，带同一份身份令牌，
所以**服务端分不出它和网页的区别**。数帧、报帧率和字节数。
一帧都没有会直接印 `NOTHING ARRIVED -- the page would have shown a blank screen`。

为什么需要它：这台 VM 开不了浏览器（`cloud-run-proxy` 组件装不上），
而**演示当天崩的三个 bug 全都只在「有人正开着大屏」时才触发**——
没人看的 e2e 恰好避开了唯一要紧的那个场景。

## 5. 在笔记本上真的用眼睛看大屏

```bash
gcloud run services proxy cctv-monitor --project=study-project-496907 \
  --region=us-central1 --port=9090
```

然后开 **`http://localhost:9090/?job=<单号>`**。

**`?job=` 不能省。** 不带就是进了空房间，画面不会推给你，
看起来跟大屏坏了一模一样。
8080 被占就换 `--port`，本地端口用哪个都不影响。

## 6. 查一单现在什么状态（Firestore）

```bash
.venv/bin/python - <<'PY'
from google.cloud import firestore
import google.auth
c,_=google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
db=firestore.Client(project="study-project-496907", database="cctv-audit", credentials=c)
for u in db.collection("cctv_audit_users").list_documents():
    for d in u.collection("jobs").list_documents():
        j=d.get().to_dict() or {}
        if j: print(d.id, u.id, j.get("state"), j.get("progress"), str(j.get("error"))[:60])
PY
```

判断卡死的三个信号，缺一不可：
`state` 还是 `running` ＋ `updated_at` 等于 `started_at`（**从没写过进度**）
＋ GCS 里 `audits/<单号>/` 只有一张 `cover.jpg`。

## 7. 看容器日志

```bash
.venv/bin/python - <<'PY'
import json,urllib.request,google.auth,google.auth.transport.requests as tr
c,_=google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"]); c.refresh(tr.Request())
body={"resourceNames":["projects/study-project-496907"],
 "filter":'resource.type="aiplatform.googleapis.com/ReasoningEngine" timestamp>="2026-09-04T05:00:00Z"',
 "orderBy":"timestamp asc","pageSize":1000}
r=urllib.request.Request("https://logging.googleapis.com/v2/entries:list",
   data=json.dumps(body).encode(), method="POST")
r.add_header("Authorization",f"Bearer {c.token}"); r.add_header("Content-Type","application/json")
for e in json.loads(urllib.request.urlopen(r).read()).get("entries",[]):
    m=" ".join((e.get("textPayload") or json.dumps(e.get("jsonPayload") or {})).split())
    print(e.get("timestamp","")[11:19], m[:170])
PY
```

> ~~**已知缺陷**：只看得到 uvicorn 的访问日志。~~
> **2026-09-08 修了**（`computer_use_agent/logsetup.py`）。以前根本没有 root handler，
> Python 的兜底 handler 只放 WARNING 以上过，所以**跑成功的那些单子也一样没日志**——
> 「没看到稽核日志」从来就不是「稽核死了」的证据。
>
> 现在容器里的日志是**一行一个 JSON、走 stdout**。两个细节是刻意的：
> stdout 而不是 stderr，因为 Cloud Run 把 stderr 上的东西一律记成 ERROR，
> 那样每条 INFO 都是红的；JSON 里带 `severity` 键，Cloud Logging 才会解析成
> 结构化条目，否则是 DEFAULT 级别，`severity>=INFO` 这种过滤一条都匹配不上。
>
> 现在该看得到的几行（认这几个字符串就能捞出一单的全过程）：
>
> ```
> Job <单号> starting: target=... start=... duration=... sop=...
> Job <单号> alive: 75s elapsed, 6 windows, 2s since the last one (allowance 300s)
> preview <单号>: pump 11.9 fps, sent 11.9 fps (179 of 179, 0 dropped by the
>   in-flight cap of 4), POST median 6ms p90 7ms, 78.5 KB/frame
> Job <单号> done in 99.8s: 10 windows, 3 violations, mode=stream
> Job <单号> killed by the watchdog: <人话原因>
> ```
>
> 本地想看人能读的格式：`LOG_FORMAT=text`。

## 8. 改完代码上云（约 15 分钟）

三步，顺序不能反。**引擎和大屏必须同一个 tag**，否则会出现
「大屏比引擎旧五个版本、一天的修改全没上线」那种事。

**现在两边都在 `agent:v18`。** 下一版把下面的 `v18` 换成新 tag：

```bash
# ① build（约 3–4 分钟，会等到 build 真结束）
.venv/bin/python deploy/agent_runtime/build_image.py v18

# ② 引擎。注意这两个脚本的参数形状不一样：
#    build_image.py 收 tag；deploy.py 收**引擎资源名**，镜像从 ENGINE_IMAGE 读。
#    直接 `deploy.py update-image v18` 会 PATCH 到 /v1/v18 上，回一个 HTML 404。
ENGINE_IMAGE=us-central1-docker.pkg.dev/study-project-496907/cctv-audit/agent:v18 \
.venv/bin/python deploy/agent_runtime/deploy.py update-image \
  projects/596821501265/locations/us-central1/reasoningEngines/6844158066963775488

# ③ 大屏（同一个 tag！它会去 Artifact Registry 核 digest，不是看 tag 名）
.venv/bin/python deploy/dashboard/deploy.py v18
```

> **部完拿 `poll` 看版本，`deploy.py` 没有 `get` 这个子命令**
> （它不报错，只是什么都不打印，看起来像在等）。

> **只改了代码就用 `update-image`。** 改了环境变量、并发数或者探针就得用
> `update-spec`——它会连 `deployment_spec` 一起重下发，而那里面有
> `MONITOR_URL` / `OIDC_ORIGINS` / `KEEPALIVE_HOLD_SECONDS` /
> `containerConcurrency` / `keepAliveProbe`。
> **v17 那次改动三样都动了，用 `update-image` 部等于白部**：
> 镜像换了，但容器还是拿不到 CPU，症状和没改一模一样。
> 部完想确认探针真的上去了，`deploy.py poll <资源名>` 看回包里有没有
> `keepAliveProbe` 和 `containerConcurrency: 2`。

`deploy.py` 其它动作：`create` / `update-methods` / `update-spec` / `poll` /
`list` / `delete <资源名>`。

> **`deploy.py` 不带参数不会有任何动作**（会打印用法）。
> 以前 `create` 是默认动作，我就是这么误建了一个引擎。

## 9. 云上真 e2e

部完之后，一边跑一边挂着观看端：

```bash
# 一个窗口：接大屏
.venv/bin/python deploy/dashboard/watch.py <单号> --seconds 600
# 另一个窗口：第 6 条那段 Firestore 查询，隔一会儿跑一次
```

**验收标准写死在这里**：必须用**演示当天你会粘的那个网址**跑，
不能用 `deploy/demovideo` 那段合成素材。合成素材跑绿只证明靶子没问题，
而且它每个窗口都判 `CANNOT_DETERMINE`（里面没有可判定的东西）。

下面每一条都要看，缺一条都不算验收通过：

| 看什么 | 在哪看 | 什么算通过 |
| :--- | :--- | :--- |
| 单子跑完了 | 第 6 条的 Firestore 查询 | `state=done`，不是卡在 `running` |
| 模型推理延迟 | 容器日志里每个窗口一行 | 单窗口 10 秒上下（v17 云上演示素材中位 8.5s；bilibili 上是 19.9s，素材信息量不一样） |
| 大屏不卡 | `watch.py` 结尾的 `freezes over 1s` | **0**；帧间隔中位贴近 `1000/PREVIEW_FPS` |
| **画面真的在动** | `/tmp/bili_v18.py` 的 `distinct pictures` / `longest identical run` | 不重样的占绝大多数；最长静止 **1 秒上下**，超过 10 秒就是那个 bug |
| **没人喊画面停了** | 容器日志 | 没有 `Dashboard picture frozen`，没有 `Chromium sent no screencast frame` |
| 容器有没有 CPU | 容器日志的 `sched: ...` 行 | `runqueue wait 0.0%`，`thread tick median 83ms` |

> `watch.py` 只数帧和字节，**不比对画面内容**，所以它看不出冻结。
> 冻结这一项现在只有 `/tmp/bili_v18.py` 能看。要在云上验，得把那两行搬进 `watch.py`。

> **平均帧率会骗人，帧间隔不会。**「一秒十二帧」和「一秒二十四帧、然后停两秒」
> 平均值一样，但只有一个能看。`watch.py` 现在同时印中位/p90/最坏间隔和
> 「超过 1 秒的卡顿次数」——**最后那个数就是操作的人抱怨的那件事本身。**

> **本地的大屏数字证明不了云上的。** 本地那 6ms 的 POST 延迟是因为大屏就在同一台机器上；
> 云上的大屏是几百毫秒外的一个 Cloud Run 服务，被投诉的 0.8–1.0 fps 幻灯片就是在那边量到的。
> 所以要认的是**容器日志里的那行 `preview ...`**，那是从容器自己的角度量的。

---

## 当前已知、待修

~~1. 容器没日志~~ / ~~2. 没有看门狗~~ / ~~3. `turn.py` 没提时间就默默跑整部片子~~
—— **三条都在 2026-09-08 修了**，各自带单测（328 → 346）：

1. **日志**：`logsetup.py`，见第 7 条。
2. **看门狗**：`audit_service.py:_watch`。它在 `pipeline.run()` **外面**，
   这是必须的——流水线自己那个 3600 秒预算只在两个窗口之间才检查
   （`pipeline.py:864`），采集环卡住就永远走不到检查点，所以那 62 分钟的单子
   本来就不可能被它救下来。三条线，都可用环境变量调：
   `AUDIT_FIRST_WINDOW_SECONDS`（首个窗口的宽限，默认 420）、
   `AUDIT_STALL_SECONDS`（此后两个窗口之间的最长沉默，默认 300）、
   `AUDIT_HARD_LIMIT_SECONDS`（总墙钟上限，默认 3900）。
   触发之后单子转 `failed`，`error` 里是一句人话（为什么停、停之前跑了几个窗口、
   下一步该怎么办），不是一个堆栈。
3. **不回落默认时间段**：`turn.py` 现在多一个 `span_stated` 字段。
   整段对话没提过时间 → 反问；只说了起点没说终点 → 也反问。
   验的时候可以故意只说「稽核这个 <网址>」，它应该回一个问题而不是开跑。

~~4. bilibili 弹登录框之后大屏画面不动~~ —— **2026-09-08 修了**（16 条单测）。
Plan A 下页面只是给人看的，原来就没开页面看门狗，也没人关弹窗、没人恢复播放。
现在两条路都开看门狗，**但只有 Plan B 有权因为页面而结束稽核**——
写反了的话 bilibili 一弹框就会把一场 5 分钟的稽核在 01:19 掐掉。
复验就是第 3 条那张表的后三行。

## 大屏卡顿：根因和怎么复验（2026-09-08）

**根因是 Cloud Run 在没有请求在处理的时候不给这个实例 CPU。**
而稽核是故意甩出请求之外跑的（流式那条路上，请求里的活会在 900.0s 被 cancel）。
所以让稽核活下来的那个决定，正好就是让它挨饿的原因。

量它的工具是新增的 `computer_use_agent/cpuprobe.py`，它每 15 秒往日志上打一行：

```
sched: thread tick median 83ms p90 83ms worst 132ms (asked 83ms, 179 in 15s);
       runqueue wait 0ms (0.0% of the window); this process 4% of one core,
       switches 1 forced / 1843 voluntary
```

**认 `runqueue wait` 那个百分比**，它是「可运行但没排上 CPU」的时间占比
（`/proc/self/schedstat` 第二个字段）。CPU 占用率那些数在这个沙箱里不可信
（它报过「0 核在跑」而活明明在干），调度等待和线程 tick 才是准的。

修之前 79–85%，修之后 0.0%。对照实验是这么做的：先让容器安静 155 秒，
再从外面猛打请求 90 秒，同一场稽核里读这一行——等待从 82% 掉到 30%，
抓帧泵从 2.7 fps 涨到 8.2 fps，Chromium 投屏从 0.0 涨到 6.6 fps。
**只差「外面有没有人在敲门」。**

修法在 `server.py` 的 `/is_busy` + `deploy.py` 里的 `keepAliveProbe`：
稽核跑着的时候这个端点**把响应挂住不返回**，于是「有请求在飞」是构造出来的。
挂多久由 `KEEPALIVE_HOLD_SECONDS` 控制。

**想做对照组就把它设成 0**（`update-spec` 下发），端点立刻返回，
探针还在但不挂请求——如果卡顿回来了，说明起作用的是「挂住」这件事本身。

一个连带约束：**挂住的探针占一个并发槽**，所以 `containerConcurrency` 必须 ≥ 2，
否则 GE 的正常轮次会吃 429。单测里有一条直接断言这个（`tests/test_container.py`）。
