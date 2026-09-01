# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Browser Use Agent Loop implementation with CDP live screencast and real-time monitoring."""

import asyncio
import os
import subprocess
import logging
import certifi
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

# Setup SSL certificates for macOS Python 3.13
if "SSL_CERT_FILE" not in os.environ:
    os.environ["SSL_CERT_FILE"] = certifi.where()
if "REQUESTS_CA_BUNDLE" not in os.environ:
    os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# Load environment variables from .env if present
_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_path):
    with open(_env_path, "r", encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

import re
from google.auth.credentials import Credentials as BaseCredentials
import google.auth
import google.auth.transport.requests
from google import genai
from google.genai.types import (
    Content,
    Part,
    FunctionResponse,
    FunctionResponsePart,
    FunctionResponseBlob,
    GenerateContentConfig,
    Tool,
    ComputerUse,
    Environment,
    FunctionDeclaration,
)
from playwright.async_api import async_playwright
from .monitor import monitor

logger = logging.getLogger("google_adk.computer_use_agent.loop")

# Custom tool for recording CCTV video segment analysis and pushing to live monitor right sidebar
record_video_segment_tool = FunctionDeclaration(
    name="record_video_segment",
    description=(
        "实时向右侧监控大屏记录一段视频的画面描述与 SOP 合规抽检结果。"
        "在每观察或分析一段视频（如 00:00 - 00:15）后必须调用此工具。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "time_range": {
                "type": "string",
                "description": "视频时间戳区间，格式如 '00:00 - 00:15' 或 '01:10 - 01:25'",
            },
            "description": {
                "type": "string",
                "description": "画面客观详细描述（人员着装、动作、操作台面、原料与器具使用状态等）",
            },
            "sop_status": {
                "type": "string",
                "enum": ["COMPLIANT", "VIOLATION", "CANNOT_DETERMINE"],
                "description": "SOP合规判定：COMPLIANT（符合标准）、VIOLATION（不符合/违规）、CANNOT_DETERMINE（因遮挡等无法判定）",
            },
            "sop_violation_detail": {
                "type": "string",
                "description": "若违规，指出具体违反的 SOP 项及现场视觉事实证据；若合规请留空",
            },
            "severity": {
                "type": "string",
                "enum": ["RED_LINE", "NORMAL", "NONE"],
                "description": "违规严重度：RED_LINE（红线违规：食安/卫生/口罩帽脱落）、NORMAL（普通违规）、NONE（无违规）",
            },
        },
        "required": ["time_range", "description", "sop_status"],
    },
)

CCTV_AUDIT_SYSTEM_INSTRUCTION = """你是一名严格、客观的霸王茶姬（CHAGEE）门店运营合规与食品安全特聘 CCTV 视频智能稽核专家。
你的核心任务是：使用浏览器自动化操作（Computer Use）访问用户提供的监控视频平台或视频网页链接，找到视频播放器，抽检其中约 1 分钟的视频，逐小段输出画面内容描述，并重点对照 SOP 标准进行合规判定，最终输出完整的分段描述及不符合 SOP 标准的问题清单。

【视频抽检与分段执行流程】
1. 访问与启动播放：使用 navigate 访问网页后，点击播放器播放按钮或按空格键启动播放。如遇全屏/跳过广告提示可进行相应操作。
2. 逐小段抽检（每 10~15 秒为一个片段，连续抽检约 1 分钟）：
   - 可以通过 `wait(seconds=10)` 持续播放观察 10 秒后捕获截屏，或通过键盘方向右键（ArrowRight）快进快退，观察不同时间点的视频画面。
   - 对每一个时间段（例如 00:00 - 00:15, 00:15 - 00:30, 00:30 - 00:45, 00:45 - 01:00）：
     * 必须调用 `record_video_segment` 工具将该时间段的内容实时推送到监控大屏！
     * 在参数中准确填写：
       - `time_range`: 如 "00:00 - 00:15"
       - `description`: 该时间段画面内容的客观详细描述（员工动作、着装佩戴、操作台物料、使用器具等）
       - `sop_status`: "COMPLIANT"（符合）/ "VIOLATION"（违规）/ "CANNOT_DETERMINE"（因遮挡等无法判定）
       - `sop_violation_detail`: 若违规，说明具体的违规事实与违背的规范项；若符合则留空
       - `severity`: "RED_LINE"（红线违规）/ "NORMAL"（普通违规）/ "NONE"
3. 持续推进直到抽检满约 1 分钟（通常经过 4~6 个分段）。
4. 抽检完成后，结束工具调用，给出最终的文字总结。

【门店 SOP 稽核标准（依据《亚太稽核远程 CCTV 巡检业务流程》）】
- 【着装仪容规范 CHK_ATTIRE_001】：必须规范佩戴工帽（碎发全塞入帽内）、规范佩戴口罩（严禁露鼻露嘴）、穿着统一工装。
- 【手部卫生红线 CHK_HYGIENE_002】：接触非洁净物（手机、垃圾桶、地面、抹布、身体/头发）后必须洗手或更换食品级手套；严禁直接用手触碰杯口、吸管内侧或食材。
- 【器具规范使用 CHK_TEA_001】：雪克杯、量杯、巴勺使用前后必须冲淋冲洗，严禁混用造成串味或生熟交叉污染。
- 【操作台与物料 CHK_SANITATION_003】：操作台面整洁无积水，抹布摆放规范，物料容器随手加盖，原料不敞口暴露。
- 【操作纪律红线 CHK_BEHAVIOR_004】：操作期间严禁看手机、聊天嬉闹或脱岗。
（若用户在指令中有自定义的特定检查项，以用户指定的规则为最高优先级进行重点核验）

【最终总结输出格式】
抽检完成后，必须输出两部分清晰报告：
### 一、 📹 视频分段内容描述总览
| 时间区间 (几分几秒 - 几分几秒) | 画面详细内容描述 | SOP 评估状态 |
| --- | --- | --- |
| 00:00 - 00:15 | ... | ✅ 符合标准 |
...

### 二、 ⚠️ 不符合 SOP 标准的问题清单
若存在违规项，逐条清晰列出：
- **违规项 #1** [时间区间] [严重等级: 红线/普通]
  - **违反规则**：...
  - **现场视觉事实证据**：...
  - **整改与复核建议**：...
若未发现任何违规项，明确注明：“✅ 经抽检，本段 1 分钟视频操作规范，未发现违反 SOP 标准的行为。”
"""


def extract_and_record_segments_from_text(text: str) -> None:
    """
    Fallback parser: extracts segment descriptions from reasoning / candidate text
    matching patterns like:
    - [00:00 - 00:15] 画面内容：...
    - 时间段：00:10 - 00:25 ...
    - 00:00 - 00:10: ...
    and pushes them to the live monitor if not already recorded.
    """
    if not text:
        return
    pattern = re.compile(r'(\d{1,2}:\d{2}\s*[-~到至]\s*\d{1,2}:\d{2})[:：\s]*(.*?)(?=(\d{1,2}:\d{2}\s*[-~到至]\s*\d{1,2}:\d{2})|$)', re.DOTALL)
    matches = pattern.findall(text)
    for match in matches:
        time_range = match[0].strip().replace("到", "-").replace("至", "-").replace("~", "-")
        desc_raw = match[1].strip()
        if len(desc_raw) < 5:
            continue
        is_viol = any(w in desc_raw for w in ["违规", "不合规", "未戴", "未佩戴", "未洗手", "串味", "污染", "VIOLATION", "不符合"])
        is_unknown = any(w in desc_raw for w in ["无法判定", "遮挡", "模糊", "CANNOT_DETERMINE"])
        sop_status = "VIOLATION" if is_viol else ("CANNOT_DETERMINE" if is_unknown else "COMPLIANT")
        severity = "RED_LINE" if is_viol and any(rw in desc_raw for rw in ["口罩", "帽子", "手套", "红线"]) else "NORMAL"

        exists = any(s.get("time_range") == time_range for s in monitor._segments)
        if not exists:
            monitor.add_video_segment(
                time_range=time_range,
                description=desc_raw[:300],
                sop_status=sop_status,
                violation_detail=desc_raw[:300] if is_viol else "",
                severity=severity,
            )


class GCloudCredentials(BaseCredentials):
    """Google Cloud credentials that refresh automatically using gcloud CLI or fallback to ADC."""

    def __init__(self):
        super().__init__()
        self.token = None
        self.refresh(None)

    def refresh(self, request=None):
        try:
            token = subprocess.check_output(
                ["gcloud", "auth", "print-access-token"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            self.token = token
        except Exception as e:
            logger.info("gcloud auth print-access-token failed (%s); attempting standard ADC fallback.", e)
            creds, _ = google.auth.default()
            creds.refresh(request or google.auth.transport.requests.Request())
            self.token = creds.token


def get_genai_client() -> genai.Client:
    """Initializes Google GenAI Client configured for Google Cloud Vertex AI."""
    project = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT", "cs-poc-hzdu6g9fvdacmw21rd6jq89")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
    # Vertex AI Computer Use preview is only supported in 'global' region
    if location != "global":
        logger.info("Vertex AI Computer Use is supported in 'global'. Overriding location '%s' to 'global'.", location)
        location = "global"

    creds = GCloudCredentials()
    return genai.Client(
        vertexai=True,
        project=project,
        location=location,
        credentials=creds,
    )


def prune_older_screenshots(contents: List[Content], keep_latest_images: int = 2) -> None:
    """
    Prunes bulky screenshot binary payloads from older turns in `contents` history.
    Keeps only the latest `keep_latest_images` screenshots, while preserving the full text
    prompt, model action history, and function response status/urls.
    This prevents request payloads from swelling to 30-50MB+ after many turns,
    which triggers Borg global load-balancing reroutes to unsupported regions (e.g. prod-gb).
    """
    image_locations = []
    for c_idx, content in enumerate(contents):
        for p_idx, part in enumerate(content.parts):
            if hasattr(part, "inline_data") and part.inline_data and getattr(part.inline_data, "mime_type", "").startswith("image/"):
                image_locations.append((c_idx, p_idx, None))
            elif hasattr(part, "function_response") and part.function_response:
                fn_resp = part.function_response
                if hasattr(fn_resp, "parts") and fn_resp.parts:
                    for sub_idx, sub_part in enumerate(fn_resp.parts):
                        if hasattr(sub_part, "inline_data") and sub_part.inline_data:
                            image_locations.append((c_idx, p_idx, sub_idx))

    if len(image_locations) > keep_latest_images:
        to_prune = image_locations[:-keep_latest_images]
        for c_idx, p_idx, sub_idx in to_prune:
            part = contents[c_idx].parts[p_idx]
            if sub_idx is None:
                contents[c_idx].parts = [p for p in contents[c_idx].parts if p != part]
            else:
                part.function_response.parts = []


def build_progress_snapshot(initial_prompt: str, current_turn: int) -> Content:
    """
    Builds a high-level progress and working memory snapshot of the continuous audit.
    Injected at the root of the sliding window to maintain O(1) context length.
    """
    segments = monitor.state.get("video_segments", [])
    violations = monitor.state.get("sop_violations", [])
    last_seg = segments[-1]["time_range"] if segments else "初始化中"

    recent_viols_summary = ""
    if violations:
        recent_viols = violations[-3:]
        recent_viols_summary = "\n最近发现的 SOP 违规事实：\n" + "\n".join(
            f"- [{v.get('time_range', '')}] {v.get('sop_violation_detail') or v.get('description', '')}"
            for v in recent_viols
        )

    snapshot_text = (
        f"{initial_prompt}\n\n"
        f"═══════════════════════════════════════════════════════════════\n"
        f"【24小时全天候巡检·进度快照与工作记忆 (第 {current_turn} 轮)】\n"
        f"• 累计已记录视频片段数: {len(segments)} 段 (最新进度至: {last_seg})\n"
        f"• 累计发现 SOP 违规: {len(violations)} 项{recent_viols_summary}\n"
        f"• 执行指引: 浏览器保持在视频播放状态。更早历史已自动持久化至本地 Checkpoint 文件中，"
        f"当前会话处于轻量滚动窗口状态。请继续观察视频推进，每发现新片段调用 record_video_segment 上报。\n"
        f"═══════════════════════════════════════════════════════════════"
    )
    return Content(role="user", parts=[Part(text=snapshot_text)])


def compact_conversation_history(
    contents: List[Content],
    initial_prompt: str,
    current_turn: int,
    keep_last_n_turns: int = 3,
) -> List[Content]:
    """
    Compacts the conversation history using a sliding window to guarantee O(1) context length.
    Ensures that for a 24-hour continuous run, Gemini never hits the context window limit.
    Preserves strict (model function_call -> user function_response) pair integrity required by Vertex AI.
    """
    required_history_len = 1 + (2 * keep_last_n_turns)
    if len(contents) <= required_history_len:
        return contents

    # Slice the last 2 * keep_last_n_turns messages
    recent_turns = contents[-(2 * keep_last_n_turns):]

    # Prepend the newly synthesized progress snapshot turn
    snapshot_turn = build_progress_snapshot(initial_prompt, current_turn)
    compacted_contents = [snapshot_turn] + recent_turns

    logger.info(
        "Compacted conversation history: %d messages -> %d messages (Turn %d, O(1) Memory Guard Active).",
        len(contents),
        len(compacted_contents),
        current_turn,
    )
    return compacted_contents


async def generate_content_with_retry(
    client: genai.Client,
    model_id: str,
    contents: List[Content],
    config: GenerateContentConfig,
    max_retries: int = 3,
):
    """Executes generate_content with automatic exponential backoff retry for transient routing issues."""
    for attempt in range(max_retries):
        try:
            return await client.aio.models.generate_content(
                model=model_id, contents=contents, config=config
            )
        except Exception as e:
            err_str = str(e)
            is_transient = (
                "computer use is not supported for this model in this region" in err_str
                or "503" in err_str
                or "429" in err_str
                or "RESOURCE_EXHAUSTED" in err_str
                or "UNAVAILABLE" in err_str
            )
            if is_transient and attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.warning(
                    "Transient Vertex AI routing/capacity error on attempt %d: %s. Retrying in %ds...",
                    attempt + 1,
                    err_str[:120],
                    wait_time,
                )
                await asyncio.sleep(wait_time)
                continue
            raise


# Screen coordinates normalization (0-1000 range to actual screen pixels)
def normalize_x(x: int, screen_width: int) -> int:
    """Convert normalized x coordinate (0-1000) to actual pixel coordinate."""
    return int(x / 1000 * screen_width)


def normalize_y(y: int, screen_height: int) -> int:
    """Convert normalized y coordinate (0-1000) to actual pixel coordinate."""
    return int(y / 1000 * screen_height)


async def execute_function_calls(
    response, page, screen_width: int, screen_height: int
) -> Tuple[str, List[Tuple[str, Any]]]:
    """Extract and execute function calls from Gemini Computer Use response."""
    candidate = response.candidates[0]
    function_calls = []
    thoughts = []

    for part in candidate.content.parts:
        if hasattr(part, "function_call") and part.function_call:
            function_calls.append(part.function_call)
        elif hasattr(part, "text") and part.text:
            thoughts.append(part.text)

    if thoughts:
        print(f"Model Reasoning: {' '.join(thoughts)}")

    if not function_calls:
        return "NO_ACTION", []

    results = []
    for function_call in function_calls:
        name = function_call.name
        args = function_call.args or {}
        print(f"Executing {name}: {args}")
        result = "success"

        norm_x = args.get("x")
        norm_y = args.get("y")

        try:
            if name == "open_web_browser":
                result = "success"

            elif name in ("navigate", "goto"):
                url = args["url"]
                await page.goto(url, timeout=60000, wait_until="domcontentloaded")
                result = "success"

            elif name in ("click_at", "click"):
                actual_x = normalize_x(args["x"], screen_width)
                actual_y = normalize_y(args["y"], screen_height)
                await page.mouse.click(actual_x, actual_y)
                result = "success"

            elif name in ("move_cursor", "move", "hover_at", "hover"):
                actual_x = normalize_x(args["x"], screen_width)
                actual_y = normalize_y(args["y"], screen_height)
                await page.mouse.move(actual_x, actual_y)
                result = "success"

            elif name == "mouse_down":
                button = args.get("button", "left")
                await page.mouse.down(button=button)
                result = "success"

            elif name == "mouse_up":
                button = args.get("button", "left")
                await page.mouse.up(button=button)
                result = "success"

            elif name in ("context_click", "right_click"):
                actual_x = normalize_x(args["x"], screen_width)
                actual_y = normalize_y(args["y"], screen_height)
                await page.mouse.click(actual_x, actual_y, button="right")
                result = "success"

            elif name in ("double_click_at", "double_click"):
                actual_x = normalize_x(args["x"], screen_width)
                actual_y = normalize_y(args["y"], screen_height)
                await page.mouse.dblclick(actual_x, actual_y)
                result = "success"

            elif name in ("triple_click_at", "triple_click"):
                actual_x = normalize_x(args["x"], screen_width)
                actual_y = normalize_y(args["y"], screen_height)
                await page.mouse.click(actual_x, actual_y, click_count=3)
                result = "success"

            elif name in ("middle_click_at", "middle_click"):
                actual_x = normalize_x(args["x"], screen_width)
                actual_y = normalize_y(args["y"], screen_height)
                await page.mouse.click(actual_x, actual_y, button="middle")
                result = "success"

            elif name in ("type_text_at", "type_text", "type"):
                if "x" in args and "y" in args:
                    actual_x = normalize_x(args["x"], screen_width)
                    actual_y = normalize_y(args["y"], screen_height)
                    await page.mouse.click(actual_x, actual_y)
                    await asyncio.sleep(0.1)

                if args.get("clear_before_typing", True):
                    await page.keyboard.press("ControlOrMeta+A")
                    await page.keyboard.press("Backspace")

                text = args.get("text", "")
                await page.keyboard.type(text)

                if args.get("press_enter", False):
                    await page.keyboard.press("Enter")
                result = "success"

            elif name in ("drag_and_drop", "drag"):
                start_x = normalize_x(args["start_x"], screen_width)
                start_y = normalize_y(args["start_y"], screen_height)
                end_x = normalize_x(args["end_x"], screen_width)
                end_y = normalize_y(args["end_y"], screen_height)
                norm_x, norm_y = args["start_x"], args["start_y"]
                await page.mouse.move(start_x, start_y)
                await page.mouse.down()
                await page.mouse.move(end_x, end_y)
                await page.mouse.up()
                result = "success"

            elif name in ("scroll_at", "scroll"):
                actual_x = normalize_x(args.get("x", 500), screen_width)
                actual_y = normalize_y(args.get("y", 500), screen_height)
                delta_x = args.get("delta_x", 0)
                delta_y = args.get("delta_y", 0)
                direction = args.get("direction", "")
                if direction == "down" and delta_y == 0:
                    delta_y = 500
                elif direction == "up" and delta_y == 0:
                    delta_y = -500
                await page.mouse.move(actual_x, actual_y)
                await page.mouse.wheel(delta_x, delta_y)
                result = "success"

            elif name == "scroll_document":
                direction = args.get("direction", "down")
                if direction == "down":
                    await page.evaluate("window.scrollBy(0, window.innerHeight * 0.8)")
                elif direction == "up":
                    await page.evaluate("window.scrollBy(0, -window.innerHeight * 0.8)")
                elif direction == "left":
                    await page.evaluate("window.scrollBy(-window.innerWidth * 0.8, 0)")
                elif direction == "right":
                    await page.evaluate("window.scrollBy(window.innerWidth * 0.8, 0)")
                result = "success"

            elif name in ("wait", "sleep"):
                seconds = min(args.get("seconds", 2), 60)
                await asyncio.sleep(seconds)
                result = "success"

            elif name in ("key_combination", "press_key", "key"):
                keys = args.get("keys") or args.get("key", "Space")
                await page.keyboard.press(keys)
                result = "success"

            elif name == "key_down":
                await page.keyboard.down(args["key"])
                result = "success"

            elif name == "key_up":
                await page.keyboard.up(args["key"])
                result = "success"

            elif name in ("go_back", "back"):
                await page.go_back()
                result = "success"

            elif name in ("go_forward", "forward"):
                await page.go_forward()
                result = "success"

            elif name in ("record_video_segment", "audit_segment"):
                time_range = args.get("time_range", "")
                description = args.get("description", "")
                sop_status = args.get("sop_status", "COMPLIANT")
                violation_detail = args.get("sop_violation_detail", args.get("violation_detail", ""))
                severity = args.get("severity", "NORMAL" if sop_status == "VIOLATION" else "NONE")
                monitor.add_video_segment(
                    time_range=time_range,
                    description=description,
                    sop_status=sop_status,
                    violation_detail=violation_detail,
                    severity=severity,
                )
                result = f"Successfully recorded segment [{time_range}] to CCTV live monitor"

            else:
                print(f"Unrecognized function: {name}")
                result = "unknown_function"

        except Exception as e:
            print(f"Error executing {name}: {e}")
            result = f"error: {str(e)}"

        # Broadcast action and click target to live monitor
        monitor.record_action(
            name=name,
            args=args,
            norm_x=norm_x,
            norm_y=norm_y,
            url=page.url,
        )

        results.append((name, result))

    return "SUCCESS", results


async def agent_loop_stream(
    initial_prompt: str,
    max_turns: int = 100,
) -> AsyncGenerator[Tuple[str, Any], None]:
    """Streams the execution of the agent loop turn-by-turn with 24h memory protection and CDP live screencast."""
    # Allow 24h continuous execution when configured
    configured_max_turns = int(os.getenv("MAX_AUDIT_TURNS", "100000"))
    if max_turns == 100 and configured_max_turns > 100:
        max_turns = configured_max_turns

    client = get_genai_client()
    model_id = os.getenv("ADK_MODEL", "gemini-3.5-flash")
    if model_id == "gemini-3.8-flash" or not model_id:
        logger.warning("Invalid model '%s' detected; defaulting to 'gemini-3.5-flash'.", model_id)
        model_id = "gemini-3.5-flash"

    config = GenerateContentConfig(
        system_instruction=CCTV_AUDIT_SYSTEM_INSTRUCTION,
        tools=[
            Tool(computer_use=ComputerUse(environment=Environment.ENVIRONMENT_BROWSER)),
            Tool(function_declarations=[record_video_segment_tool]),
        ],
    )

    sw = int(os.getenv("SCREEN_WIDTH", "1920"))
    sh = int(os.getenv("SCREEN_HEIGHT", "1080"))
    headless = os.getenv("BROWSER_HEADLESS", "true").lower() in ("true", "1", "yes")

    # Start live monitor session
    monitor.start_session(initial_prompt, max_turns=max_turns)

    playwright_loop = await async_playwright().start()
    browser_loop = await playwright_loop.chromium.launch(
        headless=headless,
        args=[
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-dev-shm-usage",
            "--autoplay-policy=no-user-gesture-required",
            "--mute-audio",
            "--js-flags=--max-old-space-size=512",
            "--disk-cache-size=104857600",
            "--disable-background-networking",
            "--disable-breakpad",
            "--disable-component-update",
        ],
    )
    page_loop = await browser_loop.new_page()
    await page_loop.set_viewport_size({"width": sw, "height": sh})

    print(f"Starting 24h Continuous Agent Loop with prompt: '{initial_prompt}' (Viewport: {sw}x{sh}, Headless: {headless})")

    # Start CDP live screencast for real-time video/screen frame streaming at full framerate & quality
    cdp = None
    try:
        cdp = await page_loop.context.new_cdp_session(page_loop)
        async def on_screencast_frame(event):
            monitor.update_frame_b64(event["data"])
            try:
                await cdp.send("Page.screencastFrameAck", {"sessionId": event["sessionId"]})
            except Exception:
                pass

        cdp.on("Page.screencastFrame", on_screencast_frame)
        await cdp.send("Page.startScreencast", {
            "format": "jpeg",
            "quality": 80,
            "maxWidth": sw,
            "maxHeight": sh,
            "everyNthFrame": 1,
        })
        logger.info("High-fidelity CDP live screencast started at full native framerate.")
    except Exception as cdp_err:
        logger.warning("CDP screencast initialization error (fallback to screenshot streaming): %s", cdp_err)

    final_text = ""
    current_url = "about:blank"
    try:
        screenshot = await page_loop.screenshot()
        current_url = page_loop.url or "about:blank"
        monitor.update_frame_bytes(screenshot, url=current_url)

        contents = [
            Content(
                role="user",
                parts=[
                    Part(text=initial_prompt),
                    Part.from_bytes(data=screenshot, mime_type="image/png"),
                ],
            )
        ]

        for turn in range(max_turns):
            print(f"\n Turn {turn + 1}")

            # 1. Periodic JavaScript Heap GC invocation every 25 turns to prevent memory bloat
            if (turn + 1) % 25 == 0:
                try:
                    await page_loop.evaluate("() => { if (window.gc) window.gc(); }")
                    logger.info("Triggered Chromium V8 garbage collection at turn %d.", turn + 1)
                except Exception:
                    pass

            # 2. Self-healing: verify page is alive; recreate if crashed
            if page_loop.is_closed():
                logger.warning("Target page was unexpectedly closed! Recreating page for 24h continuity...")
                page_loop = await browser_loop.new_page()
                await page_loop.set_viewport_size({"width": sw, "height": sh})
                if current_url and current_url != "about:blank":
                    try:
                        await page_loop.goto(current_url, timeout=30000)
                    except Exception as nav_err:
                        logger.warning("Failed to recover navigation to %s: %s", current_url, nav_err)

            # 3. O(1) Sliding Window Compaction: prevents LLM context token overflow
            contents = compact_conversation_history(
                contents=contents,
                initial_prompt=initial_prompt,
                current_turn=turn + 1,
                keep_last_n_turns=3,
            )

            # 4. Prune older screenshots to keep request lightweight and prevent global load-balancer reroutes
            prune_older_screenshots(contents, keep_latest_images=2)

            response = await generate_content_with_retry(
                client=client, model_id=model_id, contents=contents, config=config
            )

            if not response.candidates:
                print("Model returned no candidates. This may be due to a safety filter.")
                print("Full Response:", response)
                print("Terminating loop.")
                final_text = "抱歉，模型未返回可用结果或触发了安全过滤。"
                break

            candidate_content = response.candidates[0].content
            contents.append(candidate_content)

            reasoning = "".join(
                part.text
                for part in candidate_content.parts
                if hasattr(part, "text") and part.text
            )

            function_calls = [
                part.function_call
                for part in candidate_content.parts
                if hasattr(part, "function_call") and part.function_call
            ]

            # Update live monitor turn and reasoning
            monitor.set_turn_info(turn + 1, reasoning, function_calls)

            # Extract any segment descriptions in reasoning and broadcast to monitor
            if reasoning:
                extract_and_record_segments_from_text(reasoning)

            # Finish the agent loop if no function call in the response.
            if not function_calls:
                final_text = reasoning
                if final_text:
                    print(f"Agent Finished:\n{final_text}")
                    extract_and_record_segments_from_text(final_text)
                break

            status, execution_results = await execute_function_calls(
                response, page_loop, sw, sh
            )

            if status == "NO_ACTION":
                continue

            await asyncio.sleep(0.3)
            screenshot = await page_loop.screenshot()
            current_url = page_loop.url or "about:blank"
            monitor.update_frame_bytes(screenshot, url=current_url)

            function_response_parts = []
            for name, result in execution_results:
                function_response_parts.append(
                    Part(
                        function_response=FunctionResponse(
                            name=name,
                            response={"url": current_url, "result": result},
                            parts=[
                                FunctionResponsePart(
                                    inline_data=FunctionResponseBlob(
                                        mime_type="image/png", data=screenshot
                                    )
                                )
                            ],
                        )
                    )
                )

            contents.append(Content(role="user", parts=function_response_parts))
            print(f"State captured. History now has {len(contents)} messages.")

            yield (
                "turn",
                {
                    "turn": turn + 1,
                    "reasoning": reasoning,
                    "actions": execution_results,
                    "url": current_url,
                    "screenshot": screenshot,
                },
            )

        monitor.finish_session(final_text)

    except Exception as e:
        monitor.fail_session(str(e))
        raise

    finally:
        print("\n Agent loop finished. Closing browser and monitor sessions.")
        if cdp:
            try:
                await cdp.send("Page.stopScreencast")
            except Exception:
                pass
        try:
            await browser_loop.close()
        except Exception:
            pass
        try:
            await playwright_loop.stop()
        except Exception:
            pass
        try:
            await monitor.close()
        except Exception:
            pass

    yield ("finish", final_text)


async def agent_loop(initial_prompt: str, max_turns: int = 100) -> str:
    """Main agent loop compliant with user's verified specification."""
    final_response = ""
    async for event_type, data in agent_loop_stream(initial_prompt, max_turns=max_turns):
        if event_type == "finish":
            final_response = data
    return final_response
