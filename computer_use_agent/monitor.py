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

"""Browser Use Live Monitor Client & CCTV Video Audit Dashboard Template."""

import asyncio
import base64
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional
import aiohttp

logger = logging.getLogger("cctv_audit.monitor")

# HTML Page for the Frontend Monitor with CCTV Video Audit Right Sidebar
HTML_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CHAGEE CCTV 门店视频智能稽核监控台</title>
  <!-- No webfont. There used to be a render-blocking Google Fonts request
       here for JetBrains Mono + PingFang SC; measured, it returned only
       JetBrains Mono, because PingFang SC is an Apple system font that Google
       Fonts does not carry and silently drops. So the first paint of a
       dashboard someone is waiting on was held hostage to a third-party
       round trip for a font that was already installed locally on the only
       machines that have it. Every rule below already names a local stack. -->
  <style>
    :root {
      --bg-main: #0b0f19;
      --bg-panel: #111827;
      --bg-card: #1f2937;
      --border: #374151;
      --accent: #38bdf8;
      --accent-green: #10b981;
      --accent-yellow: #f59e0b;
      --accent-red: #ef4444;
      --accent-purple: #a855f7;
      --text: #e5e7eb;
      --text-muted: #9ca3af;
      --text-bright: #ffffff;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'PingFang SC', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background-color: var(--bg-main);
      color: var(--text);
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow: hidden;
    }

    /* Top Navigation Bar */
    header {
      background: var(--bg-panel);
      border-bottom: 1px solid var(--border);
      padding: 10px 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      z-index: 100;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 10px;
      font-weight: 700;
      font-size: 1.05rem;
      color: var(--text-bright);
    }
    .brand-icon {
      width: 28px;
      height: 28px;
      background: linear-gradient(135deg, #0284c7, #38bdf8);
      border-radius: 6px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #fff;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 20px;
      font-size: 0.75rem;
      font-weight: 600;
      letter-spacing: 0.5px;
      text-transform: uppercase;
    }
    .badge-running { background: rgba(16, 185, 129, 0.15); color: var(--accent-green); border: 1px solid var(--accent-green); }
    .badge-idle { background: rgba(245, 158, 11, 0.15); color: var(--accent-yellow); border: 1px solid var(--accent-yellow); }
    .badge-completed { background: rgba(56, 189, 248, 0.15); color: var(--accent); border: 1px solid var(--accent); }
    .badge-error { background: rgba(239, 68, 68, 0.15); color: var(--accent-red); border: 1px solid var(--accent-red); }

    .pulse-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: currentColor;
      animation: pulse 1.8s infinite;
    }
    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.4; transform: scale(1.3); }
    }

    .url-bar {
      flex: 1;
      max-width: 480px;
      background: var(--bg-card);
      border: 1px solid var(--border);
      padding: 6px 12px;
      border-radius: 6px;
      font-family: 'JetBrains Mono', ui-monospace, 'SF Mono', Menlo, Consolas, monospace;
      font-size: 0.8rem;
      color: var(--accent);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .url-text {
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .header-meta {
      display: flex;
      align-items: center;
      gap: 12px;
      font-size: 0.8rem;
      color: var(--text-muted);
    }
    .meta-item {
      display: flex;
      align-items: center;
      gap: 6px;
      background: var(--bg-card);
      padding: 4px 10px;
      border-radius: 6px;
      border: 1px solid var(--border);
    }
    .meta-val {
      color: var(--text-bright);
      font-family: 'JetBrains Mono', ui-monospace, 'SF Mono', Menlo, Consolas, monospace;
      font-weight: 600;
    }
    .badge-seg {
      background: rgba(56, 189, 248, 0.2);
      color: var(--accent);
      padding: 2px 6px;
      border-radius: 4px;
    }
    .badge-viol {
      background: rgba(239, 68, 68, 0.2);
      color: var(--accent-red);
      padding: 2px 6px;
      border-radius: 4px;
      font-weight: bold;
    }

    /* Main Container (Split Screen) */
    .container {
      display: flex;
      flex: 1;
      overflow: hidden;
      position: relative;
    }

    /* Left Screen View */
    .screen-area {
      flex: 1;
      display: flex;
      flex-direction: column;
      background: #000;
      position: relative;
      overflow: hidden;
    }
    .canvas-wrapper {
      flex: 1;
      position: relative;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      background-size: cover;
    }
    /* Fill the panel, letterboxing to keep the aspect ratio. The previous
       rule was `width:auto; height:auto; max-*:100%`, which only ever shrinks
       -- a 640x360 preview frame sat as a small rectangle in the middle of a
       1080p wall while the 1920x1080 placeholder SVG next to it filled the
       panel, so the picture appeared to get *smaller* the moment the audit
       started. Upscaling a preview frame is the whole point of a preview. */
    #browser-screen {
      width: 100%;
      height: 100%;
      object-fit: contain;
      box-shadow: 0 0 20px rgba(0,0,0,0.8);
      user-select: none;
      pointer-events: none;
    }

    /* Click Marker Radar Ripple */
    /* Sits over the video, not beside it: the empty video panel is the thing
       being explained, so the explanation has to be where the eye already is. */
    .ended-banner {
      position: absolute;
      left: 50%;
      top: 50%;
      transform: translate(-50%, -50%);
      max-width: 80%;
      padding: 14px 22px;
      border-radius: 10px;
      background: rgba(11, 15, 25, 0.88);
      border: 1px solid #334155;
      color: #cbd5e1;
      font-size: 0.95rem;
      line-height: 1.6;
      text-align: center;
      pointer-events: none;
      z-index: 60;
      display: none;
    }
    .click-marker {
      position: absolute;
      width: 48px;
      height: 48px;
      margin-left: -24px;
      margin-top: -24px;
      pointer-events: none;
      z-index: 50;
      display: none;
    }
    .click-marker .ring {
      position: absolute;
      width: 100%;
      height: 100%;
      border-radius: 50%;
      border: 3px solid var(--accent-red);
      background: rgba(239, 68, 68, 0.35);
      animation: radar-ripple 1.2s cubic-bezier(0, 0.2, 0.8, 1) infinite;
    }
    .click-marker .dot {
      position: absolute;
      top: 50%;
      left: 50%;
      width: 8px;
      height: 8px;
      background: var(--accent-red);
      border-radius: 50%;
      transform: translate(-50%, -50%);
      box-shadow: 0 0 8px #fff;
    }
    .click-marker .label {
      position: absolute;
      top: 100%;
      left: 50%;
      transform: translateX(-50%);
      margin-top: 4px;
      background: rgba(0,0,0,0.85);
      border: 1px solid var(--accent-red);
      color: #fff;
      font-size: 11px;
      padding: 2px 6px;
      border-radius: 4px;
      white-space: nowrap;
      font-family: 'JetBrains Mono', ui-monospace, 'SF Mono', Menlo, Consolas, monospace;
    }
    @keyframes radar-ripple {
      0% { transform: scale(0.3); opacity: 1; }
      100% { transform: scale(1.6); opacity: 0; }
    }

    /* HUD Action Banner Overlay at bottom of screen */
    .screen-hud {
      position: absolute;
      bottom: 16px;
      left: 20px;
      right: 20px;
      background: rgba(17, 24, 39, 0.9);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(56, 189, 248, 0.3);
      border-radius: 8px;
      padding: 10px 16px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      z-index: 60;
    }
    .hud-action {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .hud-action-tag {
      background: var(--accent-purple);
      color: #fff;
      font-family: 'JetBrains Mono', ui-monospace, 'SF Mono', Menlo, Consolas, monospace;
      font-size: 0.78rem;
      font-weight: 700;
      padding: 4px 8px;
      border-radius: 4px;
    }
    .hud-action-desc {
      color: var(--text-bright);
      font-size: 0.85rem;
      font-weight: 500;
    }

    /* Right Sidebar: Video Segments & SOP Inspection */
    .sidebar {
      width: 480px;
      background: var(--bg-panel);
      border-left: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    .tab-header {
      display: flex;
      border-bottom: 1px solid var(--border);
      background: var(--bg-card);
      overflow-x: auto;
    }
    .tab-btn {
      flex: 1;
      padding: 12px 10px;
      background: none;
      border: none;
      color: var(--text-muted);
      font-weight: 600;
      font-size: 0.82rem;
      cursor: pointer;
      border-bottom: 2px solid transparent;
      transition: all 0.2s;
      white-space: nowrap;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 4px;
    }
    .tab-btn.active {
      color: var(--accent);
      border-bottom-color: var(--accent);
      background: var(--bg-panel);
    }
    .tab-btn.tab-viol.has-viol {
      color: var(--accent-red);
      font-weight: bold;
    }

    .tab-content {
      flex: 1;
      overflow-y: auto;
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    /* Cards */
    .card {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px 14px;
    }
    .card-title {
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: var(--text-muted);
      margin-bottom: 8px;
      font-weight: 700;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .card-body {
      font-size: 0.85rem;
      line-height: 1.5;
      color: var(--text-bright);
      word-break: break-word;
      white-space: pre-wrap;
    }

    /* Segment Inspection Cards */
    .segment-card {
      background: #1e293b;
      border: 1px solid var(--border);
      border-left: 4px solid var(--accent);
      border-radius: 8px;
      padding: 12px 14px;
      display: flex;
      flex-direction: column;
      gap: 8px;
      animation: fadeIn 0.3s ease-in;
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .segment-card.status-compliant {
      border-left-color: var(--accent-green);
    }
    .segment-card.status-violation {
      border-left-color: var(--accent-red);
      background: rgba(239, 68, 68, 0.08);
      border-color: rgba(239, 68, 68, 0.3);
    }
    .segment-card.status-cannot-determine {
      border-left-color: var(--accent-yellow);
    }

    .segment-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .segment-time-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: rgba(56, 189, 248, 0.15);
      color: var(--accent);
      border: 1px solid rgba(56, 189, 248, 0.3);
      padding: 2px 8px;
      border-radius: 4px;
      font-family: 'JetBrains Mono', ui-monospace, 'SF Mono', Menlo, Consolas, monospace;
      font-weight: 700;
      font-size: 0.85rem;
    }
    .segment-status-badge {
      font-size: 0.75rem;
      font-weight: 600;
      padding: 2px 8px;
      border-radius: 12px;
    }
    .segment-status-badge.status-compliant {
      background: rgba(16, 185, 129, 0.15);
      color: var(--accent-green);
      border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .segment-status-badge.status-violation {
      background: rgba(239, 68, 68, 0.2);
      color: var(--accent-red);
      border: 1px solid rgba(239, 68, 68, 0.5);
    }
    .segment-status-badge.status-cannot-determine {
      background: rgba(245, 158, 11, 0.15);
      color: var(--accent-yellow);
      border: 1px solid rgba(245, 158, 11, 0.3);
    }

    .segment-desc {
      font-size: 0.86rem;
      line-height: 1.55;
      color: var(--text-bright);
    }
    .segment-violation-box {
      margin-top: 4px;
      background: rgba(239, 68, 68, 0.15);
      border: 1px solid rgba(239, 68, 68, 0.35);
      border-radius: 6px;
      padding: 8px 10px;
      font-size: 0.82rem;
      color: #fca5a5;
    }
    .violation-tag {
      font-weight: bold;
      color: #f87171;
    }

    /* SOP Violations Tab Cards */
    .violation-card {
      background: rgba(239, 68, 68, 0.08);
      border: 1px solid rgba(239, 68, 68, 0.3);
      border-left: 4px solid var(--accent-red);
      border-radius: 8px;
      padding: 12px 14px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .violation-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .violation-title {
      font-weight: 700;
      color: #f87171;
      font-size: 0.9rem;
    }
    .severity-badge {
      font-size: 0.72rem;
      padding: 2px 6px;
      border-radius: 4px;
      font-weight: 700;
      text-transform: uppercase;
    }
    .severity-red-line {
      background: var(--accent-red);
      color: #fff;
    }
    .severity-normal {
      background: var(--accent-yellow);
      color: #000;
    }
    .violation-fact {
      font-size: 0.84rem;
      color: var(--text);
      line-height: 1.5;
    }

    /* Actions */
    .copy-btn {
      background: var(--bg-card);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 6px 12px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 0.8rem;
      transition: all 0.2s;
    }
    .copy-btn:hover {
      background: var(--accent);
      color: #000;
    }

    /* Step Timeline List */
    .step-item {
      display: flex;
      gap: 12px;
      padding-bottom: 14px;
      border-left: 2px solid var(--border);
      padding-left: 14px;
      position: relative;
    }
    .step-item:last-child {
      border-left-color: transparent;
      padding-bottom: 0;
    }
    .step-dot {
      position: absolute;
      left: -6px;
      top: 2px;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--accent);
      border: 2px solid var(--bg-panel);
    }
    .step-content { flex: 1; }
    .step-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 4px;
    }
    .step-action-name {
      font-family: 'JetBrains Mono', ui-monospace, 'SF Mono', Menlo, Consolas, monospace;
      font-size: 0.82rem;
      font-weight: 700;
      color: var(--accent);
    }
    .step-time {
      font-size: 0.72rem;
      color: var(--text-muted);
      font-family: 'JetBrains Mono', ui-monospace, 'SF Mono', Menlo, Consolas, monospace;
    }
    .step-intent {
      font-size: 0.8rem;
      color: var(--text);
      margin-bottom: 4px;
    }
    .step-url {
      font-size: 0.72rem;
      color: var(--text-muted);
      font-family: 'JetBrains Mono', ui-monospace, 'SF Mono', Menlo, Consolas, monospace;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .empty-placeholder {
      color: var(--text-muted);
      font-size: 0.82rem;
      text-align: center;
      padding: 30px 10px;
      line-height: 1.6;
    }
  </style>
</head>
<body>

  <!-- Header -->
  <header>
    <div class="brand">
      <div class="brand-icon">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="2"></circle>
          <path d="M16.24 7.76a6 6 0 0 1 0 8.49m-8.48-.01a6 6 0 0 1 0-8.49m11.31-2.82a10 10 0 0 1 0 14.14m-14.14 0a10 10 0 0 1 0-14.14"></path>
        </svg>
      </div>
      <span>CHAGEE CCTV 门店视频智能稽核监控台</span>
      <div id="status-badge" class="badge badge-idle">
        <div class="pulse-dot"></div>
        <span id="status-text">IDLE 待命</span>
      </div>
    </div>

    <!-- Active URL -->
    <div class="url-bar">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="10"></circle>
        <line x1="2" y1="12" x2="22" y2="12"></line>
        <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1 4-10z"></path>
      </svg>
      <span id="current-url" class="url-text">about:blank</span>
    </div>

    <!-- Meta Info -->
    <div class="header-meta">
      <div class="meta-item">
        <span>已抽检分段:</span>
        <span id="segment-count" class="meta-val badge-seg">0 段</span>
      </div>
      <div class="meta-item">
        <span>SOP违规:</span>
        <span id="violation-count" class="meta-val badge-viol">0 项</span>
      </div>
    </div>
  </header>

  <!-- Human-in-the-loop gate: shown when a CAPTCHA / OTP blocks the pipeline -->
  <div id="intervention-bar" style="display:none; align-items:center; gap:14px; padding:10px 18px; background:#7c2d12; border-bottom:1px solid #ea580c; color:#fed7aa; font-size:0.88rem;">
    <span style="font-size:1.1rem;">🖐️</span>
    <div style="flex:1;">
      <strong>需要人工验证</strong> —
      <span id="intervention-reason">检测到验证码</span>
      <div style="font-size:0.78rem; opacity:0.85; margin-top:2px;">
        大屏只能看画面、不能操作浏览器。可行的做法：在自己电脑上登录一次该平台，把
        <code>storage_state</code> 放进 <code>AUTH_STATE_DIR</code> 后重跑；或设
        <code>BROWSER_HEADLESS=false</code> 在有图形界面的机器上手动完成。
        若确认画面里其实没有验证码，直接点右侧按钮继续即可。
      </div>
    </div>
    <button onclick="resolveIntervention()" style="padding:7px 16px; background:#ea580c; color:#fff; border:none; border-radius:6px; cursor:pointer; font-size:0.85rem;">
      已完成，继续
    </button>
  </div>

  <!-- Main Container -->
  <div class="container">
    <!-- Live Screen View -->
    <div class="screen-area">
      <div class="canvas-wrapper" id="canvas-container">
        <img id="browser-screen" src="data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='1920' height='1080' viewBox='0 0 1920 1080'><rect width='100%' height='100%' fill='%230b0f19'/><text x='50%' y='50%' fill='%23475569' font-size='28' font-family='sans-serif' text-anchor='middle'>等待 CCTV 视频巡检 Agent 启动并同步视频画面...</text></svg>" alt="Browser Stream">
        
        <!-- Says why the picture is not moving. Without it, a link opened after
             the audit finished shows a blank panel that reads as a broken
             dashboard. -->
        <div id="ended-banner" class="ended-banner"></div>

        <!-- Animated Click Radar -->
        <div id="click-marker" class="click-marker">
          <div class="ring"></div>
          <div class="dot"></div>
          <div id="click-label" class="label">click</div>
        </div>
      </div>

      <!-- Action HUD -->
      <div class="screen-hud">
        <div class="hud-action">
          <span id="hud-action-name" class="hud-action-tag">WAITING</span>
          <span id="hud-action-desc" class="hud-action-desc">等待在 ADK Web 下发监控视频抽检指令...</span>
        </div>
        <div style="color: var(--text-muted); font-size: 0.78rem; font-family: 'JetBrains Mono', monospace;">
          <span>Google Cloud Vertex AI • Gemini 3.5 Flash Computer Use</span>
        </div>
      </div>
    </div>

    <!-- Right Sidebar (Specialized CCTV Audit Stream) -->
    <div class="sidebar">
      <div class="tab-header">
        <button class="tab-btn active" onclick="switchTab('segments')">📹 视频分段巡检 (<span id="tab-seg-count">0</span>)</button>
        <button id="btn-tab-viol" class="tab-btn tab-viol" onclick="switchTab('violations')">⚠️ SOP违规清单 (<span id="tab-viol-count">0</span>)</button>
        <button class="tab-btn" onclick="switchTab('result')">📑 最终报告</button>
      </div>

      <!-- Tab: Video Segments (Default Active) -->
      <div id="tab-segments" class="tab-content">
        <div class="card" style="padding: 10px 14px; background: #182234; border-color: #2563eb;">
          <div style="display: flex; align-items: center; justify-content: space-between; font-size: 0.82rem;">
            <span id="audit-headline" style="font-weight: 600; color: #93c5fd;">🎯 视频抽检</span>
            <span id="audit-stats" style="font-family: 'JetBrains Mono', monospace; color: var(--text-muted);">已分析 0 个片段</span>
          </div>
          <div id="prompt-display" style="font-size: 0.78rem; color: var(--text-muted); margin-top: 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
            暂无进行中的抽检任务
          </div>
        </div>

        <div id="segment-list" style="display: flex; flex-direction: column; gap: 10px;">
          <div class="empty-placeholder">
            等待 Agent 访问视频播放页面并输出各时段画面分析...<br>
            格式：[几分几秒 - 几分几秒] 画面内容描述与 SOP 合规判定
          </div>
        </div>
      </div>

      <!-- Tab: SOP Violations -->
      <div id="tab-violations" class="tab-content" style="display: none;">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px;">
          <span style="font-size: 0.85rem; font-weight: 700; color: #f87171;">不符合 SOP 标准的问题清单</span>
          <button class="copy-btn" onclick="copyViolations()">📋 复制清单</button>
        </div>
        <div id="violation-list" style="display: flex; flex-direction: column; gap: 10px;">
          <div class="empty-placeholder">
            ✅ 暂未发现不符合 SOP 标准的违规项。
          </div>
        </div>
      </div>

      <!-- Tab: Final Result -->
      <div id="tab-result" class="tab-content" style="display: none;">
        <div class="card" style="flex: 1;">
          <div class="card-title">CCTV 抽检最终结构化总结</div>
          <div id="result-display" class="card-body">
            抽检任务执行中，完成后在此生成分段描述总览与违规清单报告...
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const imgEl = document.getElementById('browser-screen');
    const markerEl = document.getElementById('click-marker');
    const markerLabel = document.getElementById('click-label');
    const containerEl = document.getElementById('canvas-container');
    const statusBadge = document.getElementById('status-badge');
    const statusText = document.getElementById('status-text');
    const urlText = document.getElementById('current-url');
    const hudActionName = document.getElementById('hud-action-name');
    const hudActionDesc = document.getElementById('hud-action-desc');
    const promptDisplay = document.getElementById('prompt-display');
    const resultDisplay = document.getElementById('result-display');
    const segmentList = document.getElementById('segment-list');
    const violationList = document.getElementById('violation-list');
    const segmentCountEl = document.getElementById('segment-count');
    const violationCountEl = document.getElementById('violation-count');
    const tabSegCountEl = document.getElementById('tab-seg-count');
    const tabViolCountEl = document.getElementById('tab-viol-count');
    const btnTabViol = document.getElementById('btn-tab-viol');
    const auditStats = document.getElementById('audit-stats');
    const auditHeadline = document.getElementById('audit-headline');

    let currentTab = 'segments';
    let segmentsData = [];
    let violationsData = [];

    function switchTab(tabName) {
      currentTab = tabName;
      const tabNames = ['segments', 'violations', 'result'];
      document.querySelectorAll('.tab-btn').forEach((btn, i) => {
        btn.classList.toggle('active', tabNames[i] === tabName);
      });
      tabNames.forEach(name => {
        const el = document.getElementById(`tab-${name}`);
        if (el) el.style.display = name === tabName ? 'flex' : 'none';
      });
    }

    // Which audit this page is watching. The link handed back by preflight is
    // `.../?job=<job_id>`; without one we join the shared room, which is the
    // local `adk web` case. Everything the page asks for is scoped by this --
    // otherwise two people watching two audits see each other's footage.
    const JOB_ID = new URLSearchParams(window.location.search).get('job') || '';
    const JOB_QUERY = JOB_ID ? `?job=${encodeURIComponent(JOB_ID)}` : '';

    function connectWS() {
      const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${proto}//${window.location.host}/ws${JOB_QUERY}`;
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log('Connected to CCTV Monitor WebSocket');
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          handleMessage(msg);
        } catch (e) {
          console.error('WS Parse Error', e);
        }
      };

      ws.onclose = () => {
        console.log('WS disconnected, reconnecting in 2s...');
        setTimeout(connectWS, 2000);
      };
    }

    function handleMessage(msg) {
      if (msg.type === 'frame') {
        imgEl.src = `data:image/jpeg;base64,${msg.frame}`;
        if (msg.url) urlText.textContent = msg.url;
      } else if (msg.type === 'state') {
        updateState(msg.data);
      } else if (msg.type === 'action') {
        updateAction(msg.action, msg.click_target);
        if (msg.state) updateState(msg.state);
      } else if (msg.type === 'segment') {
        addSegment(msg.segment);
      } else if (msg.type === 'intervention') {
        showIntervention(msg.reason);
      } else if (msg.type === 'intervention_cleared') {
        hideIntervention();
      }
    }

    // -- Human-in-the-loop gate ------------------------------------------
    // We do not automate CAPTCHAs. When one appears the pipeline suspends and
    // asks a person to clear it in the live view, then continue.
    function showIntervention(reason) {
      const bar = document.getElementById('intervention-bar');
      if (!bar) return;
      document.getElementById('intervention-reason').textContent = reason || '需要人工完成验证';
      bar.style.display = 'flex';
    }

    function hideIntervention() {
      const bar = document.getElementById('intervention-bar');
      if (bar) bar.style.display = 'none';
    }

    async function resolveIntervention() {
      try {
        await fetch('/api/interact', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: 'resolve', job_id: JOB_ID }),
        });
        hideIntervention();
      } catch (e) {
        console.error('Could not signal the pipeline', e);
      }
    }

    // Whether this run has already stopped, and when. Kept across state
    // messages because a later partial update (a segment's `state` payload,
    // say) must not be read as "the run resumed".
    let endedAt = null;
    let agoTimer = null;

    function updateEndedBanner(s) {
      const el = document.getElementById('ended-banner');
      if (!el) return;
      if (s.status === 'RUNNING') endedAt = null;
      else if (s.finished_at) endedAt = s.finished_at;

      const over = (s.status === 'COMPLETED' || s.status === 'ERROR');
      if (!over) {
        // Stopping the timer here, not just hiding: it holds the state object
        // it was started with, so a run that starts again would have the old
        // "已结束" put back over the live picture a minute later.
        if (agoTimer) { clearInterval(agoTimer); agoTimer = null; }
        el.style.display = 'none';
        return;
      }

      // Frames are only ever live -- nothing is recorded and nothing replays.
      // Whoever opens the link afterwards sees either the last frame that
      // happened to be pushed, or the placeholder, and neither says why.
      let when = '', ago = '';
      if (endedAt) {
        // Local clock, because it is being read off a laptop next to a wall
        // clock. But the container and every log line are UTC, so on a CST
        // desk this reads eight hours away from the timestamps the same
        // person greps -- hence the relative age next to it, which is true
        // in any timezone.
        const t = new Date(endedAt * 1000);
        const pad = (n) => String(n).padStart(2, '0');
        when = `已于 ${pad(t.getHours())}:${pad(t.getMinutes())} `;
        const mins = Math.floor((Date.now() / 1000 - endedAt) / 60);
        ago = mins < 1 ? '（刚刚）' : (mins < 60 ? `（${mins} 分钟前）` : `（${Math.floor(mins / 60)} 小时前）`);
      }
      const how = s.status === 'ERROR' ? '中断' : '结束';
      el.textContent = `这场稽核${when}${how}了${ago}，没有实时画面。`
                     + '下方的巡检结果和证据截图是完整的。';
      el.style.display = 'block';
      // Nothing arrives after a run ends, so without this the "刚刚" that was
      // true when the banner appeared stays on screen for the rest of the day.
      if (!agoTimer) agoTimer = setInterval(() => updateEndedBanner(s), 60000);
    }

    function updateState(s) {
      if (!s) return;
      statusBadge.className = 'badge';
      if (s.status === 'RUNNING') {
        statusBadge.classList.add('badge-running');
        statusText.textContent = 'RUNNING 抽检中';
      } else if (s.status === 'COMPLETED') {
        statusBadge.classList.add('badge-completed');
        statusText.textContent = 'COMPLETED 完成';
      } else if (s.status === 'ERROR') {
        statusBadge.classList.add('badge-error');
        statusText.textContent = 'ERROR 异常';
      } else {
        statusBadge.classList.add('badge-idle');
        statusText.textContent = 'IDLE 待命';
      }

      if (s.prompt) promptDisplay.textContent = s.prompt;
      if (s.capture_settings !== undefined) {
        auditHeadline.textContent = s.capture_settings ? `🎯 ${s.capture_settings}` : '🎯 视频抽检';
      }
      if (s.current_url) urlText.textContent = s.current_url;
      if (s.last_action) {
        hudActionName.textContent = s.last_action;
        hudActionDesc.textContent = (s.last_action_args && s.last_action_args.intent) || '正在执行抽检动作';
      }

      if (s.final_result) {
        resultDisplay.textContent = s.final_result;
      }

      updateEndedBanner(s);

      if (s.video_segments && s.video_segments.length > 0) {
        segmentsData = s.video_segments;
        renderSegments();
      }
      if (s.sop_violations && s.sop_violations.length > 0) {
        violationsData = s.sop_violations;
        renderViolations();
      }
    }

    function addSegment(seg) {
      if (!seg) return;
      const idx = segmentsData.findIndex(s => s.time_range === seg.time_range);
      if (idx >= 0) {
        segmentsData[idx] = seg;
      } else {
        segmentsData.push(seg);
      }
      if (seg.sop_status === 'VIOLATION') {
        const vIdx = violationsData.findIndex(v => v.time_range === seg.time_range);
        if (vIdx >= 0) {
          violationsData[vIdx] = seg;
        } else {
          violationsData.push(seg);
        }
      }
      renderSegments();
      renderViolations();
    }

    function renderSegments() {
      segmentCountEl.textContent = `${segmentsData.length} 段`;
      tabSegCountEl.textContent = segmentsData.length;
      const violCount = segmentsData.filter(s => s.sop_status === 'VIOLATION').length;
      auditStats.textContent = `已分析 ${segmentsData.length} 段 | 违规 ${violCount} 项`;

      if (segmentsData.length === 0) {
        segmentList.innerHTML = '<div class="empty-placeholder">等待 Agent 访问视频播放页面并输出各时段画面分析...</div>';
        return;
      }

      segmentList.innerHTML = '';
      segmentsData.forEach(seg => {
        const div = document.createElement('div');
        const isViol = seg.sop_status === 'VIOLATION';
        const isUnknown = seg.sop_status === 'CANNOT_DETERMINE';
        const statusClass = isViol ? 'status-violation' : (isUnknown ? 'status-cannot-determine' : 'status-compliant');
        const statusText = isViol ? '❌ 违规' : (isUnknown ? '⚠️ 无法判定' : '✅ 符合SOP');

        div.className = `segment-card ${statusClass}`;
        div.innerHTML = `
          <div class="segment-header">
            <div class="segment-time-badge">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"></circle>
                <polyline points="12 6 12 12 16 14"></polyline>
              </svg>
              ${seg.time_range}
            </div>
            <div class="segment-status-badge ${statusClass}">
              ${statusText} ${seg.severity && isViol ? `[${seg.severity}]` : ''}
            </div>
          </div>
          <div class="segment-desc">
            <strong>画面内容：</strong>${seg.description || '无画面描述'}
          </div>
          ${isViol && seg.violation_detail ? `
            <div class="segment-violation-box">
              <div class="violation-tag">❌ SOP 违规事实：</div>
              <div style="margin-top: 2px;">${seg.violation_detail}</div>
            </div>
          ` : ''}
          ${renderFindings(seg)}
        `;
        segmentList.appendChild(div);
      });
      const tabEl = document.getElementById('tab-segments');
      if (tabEl) tabEl.scrollTop = tabEl.scrollHeight;
    }

    const STATUS_MARK = { VIOLATION: '❌', COMPLIANT: '✅', CANNOT_DETERMINE: '❔' };

    function escapeHtml(s) {
      return String(s == null ? '' : s).replace(/[&<>"']/g,
        c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
    }

    // Per-rule verdicts. CANNOT_DETERMINE is shown rather than hidden: "we
    // could not see it" is a real audit outcome and hiding it would make
    // coverage look better than it is.
    function renderFindings(seg) {
      if (!seg.findings || !seg.findings.length) return '';
      const rows = seg.findings.map(f => `
        <div style="display:flex; gap:6px; align-items:baseline; padding:2px 0;">
          <span>${STATUS_MARK[f.status] || '•'}</span>
          <span style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:var(--text-muted);">${escapeHtml(f.rule_id)}</span>
          <span style="flex:1; font-size:0.78rem;">${escapeHtml(f.evidence) || '—'}</span>
          <span style="font-size:0.7rem; color:var(--text-muted);">${escapeHtml(f.timestamp || '')} · ${Math.round((f.confidence || 0) * 100)}%</span>
        </div>
      `).join('');
      return `<div style="margin-top:6px; border-top:1px dashed #334155; padding-top:6px;">${rows}</div>`;
    }

    function renderEvidence(v) {
      const frames = (v.findings || [])
        .filter(f => f.status === 'VIOLATION' && f.evidence_frame)
        .map(f => `
          <a href="/api/evidence/${encodeURI(f.evidence_frame)}" target="_blank" title="${escapeHtml(f.rule_id)} @ ${escapeHtml(f.timestamp || '')}">
            <img src="/api/evidence/${encodeURI(f.evidence_frame)}"
                 style="width:132px; border-radius:5px; border:1px solid #475569;" loading="lazy">
          </a>
        `);
      if (!frames.length) return '';
      return `<div style="display:flex; gap:6px; flex-wrap:wrap; margin-top:6px;">${frames.join('')}</div>`;
    }

    function renderViolations() {
      const violCount = violationsData.length;
      violationCountEl.textContent = `${violCount} 项`;
      tabViolCountEl.textContent = violCount;
      btnTabViol.classList.toggle('has-viol', violCount > 0);

      if (violCount === 0) {
        violationList.innerHTML = '<div class="empty-placeholder">✅ 当前抽检片段未发现不符合 SOP 标准的违规行为。</div>';
        return;
      }

      violationList.innerHTML = '';
      violationsData.forEach((v, index) => {
        const div = document.createElement('div');
        div.className = 'violation-card';
        const isRedLine = v.severity === 'RED_LINE';
        div.innerHTML = `
          <div class="violation-header">
            <span class="violation-title">#${index + 1} [${v.time_range}]</span>
            <span class="severity-badge ${isRedLine ? 'severity-red-line' : 'severity-normal'}">
              ${isRedLine ? '红线违规 (RED LINE)' : '普通违规'}
            </span>
          </div>
          <div class="violation-fact">
            <strong>违规内容：</strong>${v.violation_detail || v.description}
          </div>
          <div style="font-size: 0.78rem; color: var(--text-muted); margin-top: 2px;">
            视觉证据时间点：${v.time_range} • 点击证据图可放大复核
          </div>
          ${renderEvidence(v)}
        `;
        violationList.appendChild(div);
      });
    }

    function copyViolations() {
      if (violationsData.length === 0) {
        alert('当前没有不符合SOP的违规项可复制。');
        return;
      }
      let text = `【CHAGEE CCTV 门店抽检 - 不符合 SOP 违规清单】\n共发现 ${violationsData.length} 项违规：\n\n`;
      violationsData.forEach((v, i) => {
        text += `${i + 1}. [时间段: ${v.time_range}] [级别: ${v.severity || 'NORMAL'}]\n`;
        text += `   违规事实: ${v.violation_detail || v.description}\n\n`;
      });
      navigator.clipboard.writeText(text).then(() => {
        alert('违规清单已复制到剪贴板！');
      }).catch(err => {
        console.error('Copy failed', err);
      });
    }

    function updateAction(action, clickTarget) {
      if (!action) return;
      hudActionName.textContent = action.action;
      hudActionDesc.textContent = action.intent || JSON.stringify(action.args);

      if (clickTarget && clickTarget.norm_x !== undefined && clickTarget.norm_y !== undefined) {
        showClickMarker(clickTarget.norm_x, clickTarget.norm_y, clickTarget.action);
      } else {
        markerEl.style.display = 'none';
      }
    }

    function showClickMarker(normX, normY, actionName) {
      const rect = imgEl.getBoundingClientRect();
      const containerRect = containerEl.getBoundingClientRect();

      const posX = rect.left - containerRect.left + (normX / 1000) * rect.width;
      const posY = rect.top - containerRect.top + (normY / 1000) * rect.height;

      markerEl.style.left = `${posX}px`;
      markerEl.style.top = `${posY}px`;
      markerLabel.textContent = `${actionName} (${normX}, ${normY})`;
      markerEl.style.display = 'block';

      setTimeout(() => {
        markerEl.style.display = 'none';
      }, 2500);
    }

    fetch('/api/state' + JOB_QUERY).then(r => r.json()).then(s => updateState(s)).catch(() => {});
    connectWS();
  </script>
</body>
</html>
"""


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name) or default)
    except ValueError:
        logger.warning("%s is not a whole number; using %s.", name, default)
        return default


# How many preview frames may be on the wire at once. Sized to cover the round
# trip to a Cloud Run dashboard (a few hundred ms) at 12 fps without letting a
# genuinely stalled link accumulate a backlog. See `update_frame_b64`.
#
# Tunable by environment because changing it is the one cheap experiment for a
# slideshow dashboard, and a rebuild to try a different number costs ten
# minutes while an engine env update costs one.
_MAX_FRAMES_IN_FLIGHT = _env_int("MAX_FRAMES_IN_FLIGHT", 4)

# How often the sender says out loud what it is doing to the preview stream.
#
# The dashboard has been reported as a slideshow twice, and both investigations
# stalled at the same place: PREVIEW_FPS says 12, the viewer counts 1, and
# nothing in between is observable. Dropping frames is the *designed* behaviour
# of the cap above, so it is silent by construction -- the counter it bumps was
# only ever read in tests. This makes the drop rate and the round trip visible
# from a log, which is the difference between measuring the problem and
# guessing at it again.
_FRAME_REPORT_SECONDS = 15.0


class BrowserMonitorClient:
    """Sends browser use frames, action events, and video segment audit results to the monitor server."""

    def __init__(self, job_id: str = ""):
        from .config import config as _config

        # Which dashboard room this client writes to. Empty is the shared room
        # -- `adk web`, where there is one audit and one page. In the cloud
        # every audit gets its own, so that two customers watching two audits
        # on the same service do not see each other's CCTV footage.
        self.job_id = job_id
        self.port = _config.monitor_port
        self.token = _config.monitor_token
        # One place that decides where the dashboard is. Locally it is the
        # other process on this box; on Agent Runtime it is a Cloud Run service
        # and 127.0.0.1 would post frames into the void -- silently, because
        # every send here is fire-and-forget by design.
        self.base_url = _config.monitor_url or f"http://127.0.0.1:{_config.monitor_port}"
        self._id_token = ""
        self._id_token_expiry = 0.0
        self._id_token_warned = False
        self.screen_width = _config.screen_width
        self.screen_height = _config.screen_height
        self._session: Optional[aiohttp.ClientSession] = None
        self._segments: List[Dict[str, Any]] = []
        # Frames are the only high-rate event, and the only one worth dropping.
        self._frames_in_flight = 0
        self._frames_dropped = 0
        # Everything below is for the periodic report in `_frame_report`. Kept
        # on the client rather than in a global so two audits in one container
        # do not average each other's numbers together.
        self._frames_offered = 0
        self._frames_sent = 0
        self._frame_bytes = 0
        self._send_ms: List[float] = []
        self._report_at = 0.0

    @property
    def state(self) -> Dict[str, Any]:
        return {
            "video_segments": self._segments,
            "sop_violations": [s for s in self._segments if s.get("sop_status") == "VIOLATION"],
        }

    async def close(self):
        """Closes internal aiohttp client session cleanly."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=1.5)
            )
        return self._session

    async def _headers(self) -> Dict[str, str]:
        """What every call to the dashboard carries.

        Two different things, and both are needed once the dashboard is a
        separate Cloud Run service:

          * `X-Monitor-Token` is the application's own shared secret. It is
            what `monitor_server` checks, and it is all that is needed when the
            two run on one box.
          * `Authorization` is a Google-issued identity token. The dashboard is
            deployed `--no-allow-unauthenticated` -- the org policy forbids
            `allUsers` anyway -- so Cloud Run rejects the request before
            `monitor_server` ever sees it unless one is attached.

        Fetching the identity token needs a service account, which is what runs
        in the cloud and is exactly what a workstation's user credentials are
        not. So a failure here is logged once and shrugged off: locally there
        is nothing in front of the dashboard to satisfy.
        """
        headers = {"X-Monitor-Token": self.token} if self.token else {}
        if not self.base_url.startswith("https://"):
            return headers

        token = await self._identity_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    async def _identity_token(self) -> str:
        """An OIDC token for the dashboard's URL, refreshed a minute early."""
        if self._id_token and time.time() < self._id_token_expiry - 60:
            return self._id_token

        def fetch() -> tuple:
            import google.auth.transport.requests
            from google.oauth2 import id_token as id_token_lib

            request = google.auth.transport.requests.Request()
            raw = id_token_lib.fetch_id_token(request, self.base_url)
            # The expiry is inside the token; decoding it would mean pulling in
            # a JWT library for one number. Google's are an hour, so re-fetch
            # every fifty minutes and let the metadata server cache it.
            return raw, time.time() + 3000

        try:
            self._id_token, self._id_token_expiry = await asyncio.to_thread(fetch)
        except Exception as exc:
            if not self._id_token_warned:
                self._id_token_warned = True
                logger.warning(
                    "No identity token for %s (%s). Dashboard updates will be "
                    "rejected if it requires authentication.",
                    self.base_url, str(exc)[:120],
                )
            return ""
        return self._id_token

    def _fire_and_forget(self, payload: Dict[str, Any], on_done=None):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No loop: the caller is not in async context, so nothing can be
            # sent. Release the sender rather than wedging it shut forever.
            if on_done is not None:
                on_done()
            return
        task = loop.create_task(self._post_event(payload))
        if on_done is not None:
            task.add_done_callback(lambda _t: on_done())

    async def _post_event(self, payload: Dict[str, Any]):
        try:
            session = await self._get_session()
            headers = await self._headers()
            # Stamped here rather than at every call site: forgetting it on one
            # event type would put that event in the wrong room, and the
            # symptom -- a card appearing on somebody else's dashboard -- is
            # not one anybody would connect back to a missing field.
            if self.job_id:
                payload = {**payload, "job_id": self.job_id}
            async with session.post(
                f"{self.base_url}/api/event", json=payload, headers=headers
            ):
                pass
        except Exception:
            pass

    def start_session(self, prompt: str, settings: str = ""):
        self._segments = []
        self._fire_and_forget({
            "type": "state",
            "data": {
                "status": "RUNNING",
                "prompt": prompt,
                # What this run is actually configured to do. The banner it
                # feeds used to be a hardcoded string.
                "capture_settings": settings,
                "current_url": "about:blank",
                "last_action": "启动浏览器",
                "final_result": "",
                "video_segments": [],
                "sop_violations": [],
            },
        })

    def push_record(self, record: Dict[str, Any]) -> None:
        """Renders an AuditStore record onto the dashboard.

        The store owns the record shape; this flattens it into the fields the
        sidebar renders, and passes `findings` through so per-rule verdicts and
        evidence frames can be shown.
        """
        violations = [f for f in record.get("findings", []) if f.get("status") == "VIOLATION"]
        detail = "；".join(
            f"[{f.get('rule_id')}@{f.get('timestamp', '')}] {f.get('evidence', '')}"
            for f in violations
        )
        seg = {
            "id": record.get("id"),
            "time_range": record.get("time_range", ""),
            # So the dashboard can order cards by video time. Analysis runs
            # concurrently, so arrival order is whichever window finished
            # first -- correct, but unreadable as a timeline.
            "start_seconds": record.get("start_offset_seconds", 0.0),
            "description": record.get("scene_summary", ""),
            "sop_status": record.get("sop_status", "COMPLIANT"),
            "violation_detail": detail,
            "severity": record.get("severity", "NONE"),
            "findings": record.get("findings", []),
            "people_count": record.get("people_count", 0),
            "visibility_ok": record.get("visibility_ok", True),
            "time": time.strftime("%H:%M:%S"),
        }
        self._segments.append(seg)
        self._fire_and_forget({"type": "segment", "segment": seg})

    def request_human(self, reason: str) -> None:
        """Raises the "needs a person" banner on the dashboard."""
        self._fire_and_forget({"type": "intervention", "reason": reason})

    def clear_human_request(self) -> None:
        self._fire_and_forget({"type": "intervention_cleared"})

    async def read_intervention(self) -> tuple:
        """Reads back the banner the dashboard is showing, for `HumanGate`.

        Returns `(reachable, banner_text_or_None)`. The two are separate because
        "no banner" and "no dashboard" must not be confused: the first means the
        operator pressed 继续, the second means nobody is watching.
        """
        try:
            session = await self._get_session()
            headers = await self._headers()
            async with session.get(
                f"{self.base_url}/api/state", params=self._job_params(), headers=headers
            ) as response:
                if response.status != 200:
                    return False, None
                data = await response.json()
            return True, data.get("intervention")
        except Exception:
            return False, None

    async def viewers(self) -> int:
        """How many people currently have the dashboard open.

        Used to decide whether streaming preview frames is worth anything.
        Returns 0 when the dashboard cannot be reached, which is the same
        answer as "nobody is watching" and leads to the same behaviour: send
        nothing. Erring the other way would stream a live CCTV feed at a
        service that is not there.
        """
        try:
            session = await self._get_session()
            headers = await self._headers()
            async with session.get(
                f"{self.base_url}/api/viewers", params=self._job_params(), headers=headers
            ) as response:
                if response.status != 200:
                    return 0
                return int((await response.json()).get("viewers") or 0)
        except Exception:
            return 0

    def _job_params(self) -> Dict[str, str]:
        """Scopes a read to this audit's room.

        Without it a run streams frames for as long as *anyone* has *any*
        dashboard open, which is both the wrong answer and an expensive one.
        """
        return {"job": self.job_id} if self.job_id else {}

    def note(self, action: str, detail: str = "") -> None:
        """Says what the run is doing, in the dashboard's action banner."""
        self._fire_and_forget({
            "type": "action",
            "action": {"action": action, "args": {"intent": detail} if detail else {}},
        })

    def update_frame_b64(self, b64_str: str):
        """Pushes a preview frame, dropping it if too many are already going.

        There has to be a cap: without one, a slow link turns a 12 fps preview
        into an unbounded pile of pending POST tasks that arrive late, in
        bursts, and out of order -- which looks worse on the dashboard than
        simply showing fewer frames. A dropped preview frame costs nothing; the
        evidence feed is a separate stream.

        The cap used to be one, and that quietly made the round trip the frame
        rate: the dashboard is a Cloud Run service several hundred milliseconds
        away, so 12 fps arrived as **0.5 fps** in a measured run -- a
        slideshow, whatever `PREVIEW_FPS` says. Nothing logged it, because
        dropping frames is the designed behaviour and the counter it increments
        is only read in tests.

        A few in flight covers the round trip without reintroducing the pile.
        Frames can now land out of order, which at 12 fps means the picture can
        be one frame stale for ~80ms. That is not visible; a slideshow is.
        """
        self._frames_offered += 1
        if self._report_at == 0.0:
            self._report_at = time.monotonic()
        if self._frames_in_flight >= _MAX_FRAMES_IN_FLIGHT:
            self._frames_dropped += 1
            self._frame_report()
            return
        self._frames_in_flight += 1
        started = time.monotonic()
        self._fire_and_forget(
            {"type": "frame", "frame": b64_str},
            on_done=lambda: self._frame_sent(started, len(b64_str)),
        )

    def _frame_sent(self, started: float = 0.0, size: int = 0) -> None:
        # Never below zero: `_fire_and_forget` calls this synchronously when
        # there is no event loop, and a counter that drifts negative would
        # uncap the sender instead of capping it.
        self._frames_in_flight = max(0, self._frames_in_flight - 1)
        self._frames_sent += 1
        self._frame_bytes += size
        if started:
            self._send_ms.append((time.monotonic() - started) * 1000.0)
        self._frame_report()

    def _frame_report(self) -> None:
        """Says, every 15 seconds, what actually happened to the preview.

        Four numbers, because between them they name every place a frame can go
        missing: how many the pump offered, how many went out, how many the cap
        threw away, and how long the round trip took. Reported as one line so
        that reading it later is not an exercise in correlating timestamps.
        """
        now = time.monotonic()
        window = now - self._report_at
        if window < _FRAME_REPORT_SECONDS:
            return
        self._report_at = now

        latencies = sorted(self._send_ms)
        offered, sent, dropped = self._frames_offered, self._frames_sent, self._frames_dropped
        self._frames_offered = self._frames_sent = self._frames_dropped = 0
        self._send_ms = []
        avg_kb = (self._frame_bytes / sent / 1024.0) if sent else 0.0
        self._frame_bytes = 0

        def pct(p: float) -> float:
            if not latencies:
                return 0.0
            return latencies[min(len(latencies) - 1, int(len(latencies) * p))]

        logger.info(
            "preview %s: pump %.1f fps, sent %.1f fps (%d of %d, %d dropped by "
            "the in-flight cap of %d), POST median %.0fms p90 %.0fms, "
            "%.1f KB/frame",
            self.job_id or "local",
            offered / window, sent / window, sent, offered, dropped,
            _MAX_FRAMES_IN_FLIGHT, pct(0.5), pct(0.9), avg_kb,
        )

    def update_frame_bytes(self, frame_bytes: bytes, url: Optional[str] = None):
        b64_str = base64.b64encode(frame_bytes).decode("ascii")
        self._fire_and_forget({
            "type": "frame",
            "frame": b64_str,
            "url": url,
        })

    def record_action(
        self,
        name: str,
        args: Dict[str, Any],
        norm_x: Optional[int] = None,
        norm_y: Optional[int] = None,
        url: Optional[str] = None,
    ):
        click_target = None
        if norm_x is not None and norm_y is not None:
            # Scale to the configured viewport, not a hardcoded 1080p one, or
            # the radar marker lands in the wrong place on any other size.
            click_target = {
                "x": int(norm_x * self.screen_width / 1000),
                "y": int(norm_y * self.screen_height / 1000),
                "norm_x": norm_x,
                "norm_y": norm_y,
                "action": name,
            }

        step_record = {
            "turn": 0,
            "action": name,
            "args": args,
            "url": url or "",
            "time": time.strftime("%H:%M:%S"),
            "intent": args.get("intent", ""),
        }

        self._fire_and_forget({
            "type": "action",
            "action": step_record,
            "click_target": click_target,
        })

    def finish_session(self, final_text: str):
        self._fire_and_forget({
            "type": "state",
            "data": {
                "status": "COMPLETED",
                "final_result": final_text,
                "click_target": None,
            },
        })

    def fail_session(self, error_msg: str):
        self._fire_and_forget({
            "type": "state",
            "data": {
                "status": "ERROR",
                "final_result": f"执行出错: {error_msg}",
            },
        })


# The shared-room client: `adk web` and anything else with one audit per
# process. Cloud runs build their own with `monitor_for_job`.
monitor = BrowserMonitorClient()


def monitor_for_job(job_id: str) -> BrowserMonitorClient:
    """A dashboard client scoped to one audit.

    A fresh object rather than a job id set on the singleton, because the
    container is allowed to run several audits at once -- `start_audit` returns
    immediately and the work carries on in a detached task -- and a shared
    client would mean two runs stamping each other's job id onto their frames
    and appending to each other's `_segments`. Whoever builds one closes it.
    """
    return BrowserMonitorClient(job_id=job_id)
