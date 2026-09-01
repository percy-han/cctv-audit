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

"""Computer Use Agent definition for Google ADK using Google Cloud Vertex AI Authorization."""

import os
import certifi
import logging
from typing import AsyncGenerator

# Fix macOS Python 3.13 SSL verification issues with aiohttp / Google APIs
if "SSL_CERT_FILE" not in os.environ:
    os.environ["SSL_CERT_FILE"] = certifi.where()
if "REQUESTS_CA_BUNDLE" not in os.environ:
    os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from google.adk.agents.base_agent import BaseAgent
from google.adk.events import Event
from google.adk.agents.invocation_context import InvocationContext
from google.adk.tools.computer_use.computer_use_toolset import ComputerUseToolset
from google.genai import types

from .computer import PlaywrightComputer
from .agent_loop import agent_loop, agent_loop_stream

logger = logging.getLogger("google_adk.computer_use_agent")

SCREEN_WIDTH = int(os.getenv("SCREEN_WIDTH", "1920"))
SCREEN_HEIGHT = int(os.getenv("SCREEN_HEIGHT", "1080"))
ALLOW_PRIVATE_NETWORK = os.getenv("ALLOW_PRIVATE_NETWORK_ACCESS", "true").lower() in ("true", "1", "yes")

# Playwright computer instance for ADK metadata & tools
computer = PlaywrightComputer(
    width=SCREEN_WIDTH,
    height=SCREEN_HEIGHT,
)

computer_toolset = ComputerUseToolset(
    computer=computer,
    allow_private_network_access=ALLOW_PRIVATE_NETWORK,
)


class BrowserUseAgent(BaseAgent):
    """ADK Agent specialized for CCTV store video audit via Gemini Computer Use with live segment monitor."""

    # Registering tools allows ADK Web to identify this as a Computer Use Agent
    tools: list = [computer_toolset]

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """Runs the CCTV browser audit loop upon receiving a prompt or URL from the user in ADK Web."""
        user_prompt = ""
        if ctx.user_content and ctx.user_content.parts:
            user_prompt = "".join(
                p.text or "" for p in ctx.user_content.parts if hasattr(p, "text")
            )

        if not user_prompt.strip():
            yield Event(
                invocation_id=ctx.invocation_id,
                author=self.name,
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_text(text="请提供待抽检的监控视频平台或视频网页链接（例如 Bilibili 或监控平台 URL），并可附带具体的 SOP 检查关注项。")],
                ),
            )
            return

        # Prepare specialized CCTV audit prompt wrapper
        prompt = user_prompt.strip()
        if not any(k in prompt for k in ("抽检", "SOP", "分段", "几分几秒")):
            prompt += (
                "\n\n【稽核任务要求】\n"
                "请使用浏览器访问该页面并定位视频播放器，启动视频播放并抽检其中约 1 分钟的内容。\n"
                "每 10~15 秒为一个片段，调用 record_video_segment 工具记录画面内容（几分几秒到几分几秒，画面中的内容是什么），"
                "并重点检查是否符合门店 SOP 标准（工装/口罩/帽子、手部卫生/手套、器具清洁、台面卫生等）。\n"
                "最终输出完整的分段描述总览表，以及一份不符合 SOP 标准的问题清单。"
            )

        print(f"[CCTV Audit Agent] Received prompt: {prompt}")

        async for event_type, data in agent_loop_stream(prompt):
            if event_type == "turn":
                turn_num = data["turn"]
                reasoning = data.get("reasoning", "").strip()
                actions = data.get("actions", [])
                action_names = ", ".join(f"`{name}`" for name, _ in actions) if actions else "观察视频画面"
                url = data.get("url", "")

                status_text = f"🔄 **[Turn {turn_num}]** 执行动作: {action_names} | 页面: {url}"
                if reasoning:
                    status_text += f"\n> {reasoning[:200]}"

                yield Event(
                    invocation_id=ctx.invocation_id,
                    author=self.name,
                    content=types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=status_text)],
                    ),
                )

            elif event_type == "finish":
                final_text = data or "CCTV 视频抽检已执行完毕。"
                yield Event(
                    invocation_id=ctx.invocation_id,
                    author=self.name,
                    content=types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=final_text)],
                    ),
                )


root_agent = BrowserUseAgent(
    name="computer_use_agent",
    description="专用 AI 门店远程 CCTV 视频智能抽检 Agent，基于 Google Cloud Vertex AI 访问监控页面并抽检 1 分钟视频，右侧大屏实时呈现分段画面描述与不符合 SOP 的问题清单。",
)
