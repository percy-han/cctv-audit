"""Standalone test script to verify agent_loop and ADK root_agent with Google Cloud Vertex AI."""

import asyncio
import os
import sys

# Add parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from computer_use_agent import root_agent, agent_loop
from google.adk.runners import InMemoryRunner
from google.genai import types


async def test_agent_loop():
    print("\n--- 1. Testing agent_loop directly (User Verified Pattern) ---")
    prompt = "访问 https://example.com 并提取页面上的主标题内容"
    print(f"  Prompt: '{prompt}'")
    result = await agent_loop(prompt, max_turns=5)
    print(f"\n  Result from agent_loop:\n{result}\n")
    assert result, "agent_loop should return non-empty text"
    print("  agent_loop test passed!")


async def test_adk_root_agent():
    print("\n--- 2. Testing ADK root_agent via InMemoryRunner ---")
    runner = InMemoryRunner(agent=root_agent)
    session = await runner.session_service.create_session(app_name=runner.app_name, user_id="test_user")
    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text="访问 https://example.com ，告诉我页面标题")]
    )
    step = 0
    async for event in runner.run_async(session_id=session.id, user_id="test_user", new_message=content):
        step += 1
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(f"  [Step {step}] Event: {part.text[:120]}")
    print("  ADK root_agent test passed!")


async def main():
    print("============================================================")
    print("Testing Browser Use Agent (Google Cloud Vertex AI)")
    print("============================================================")
    await test_agent_loop()
    await test_adk_root_agent()
    print("\n🎉 All tests passed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
