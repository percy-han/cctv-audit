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
import os
import time
from typing import Any, Dict, List, Optional
import aiohttp

# HTML Page for the Frontend Monitor with CCTV Video Audit Right Sidebar
HTML_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CHAGEE CCTV 门店视频智能稽核监控台</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=PingFang+SC:wght@400;500;600;700&display=swap" rel="stylesheet">
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
      font-family: 'JetBrains Mono', monospace;
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
      font-family: 'JetBrains Mono', monospace;
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
    #browser-screen {
      max-width: 100%;
      max-height: 100%;
      width: auto;
      height: auto;
      object-fit: contain;
      box-shadow: 0 0 20px rgba(0,0,0,0.8);
      user-select: none;
      pointer-events: none;
    }

    /* Click Marker Radar Ripple */
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
      font-family: 'JetBrains Mono', monospace;
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
      font-family: 'JetBrains Mono', monospace;
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
      font-family: 'JetBrains Mono', monospace;
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
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.82rem;
      font-weight: 700;
      color: var(--accent);
    }
    .step-time {
      font-size: 0.72rem;
      color: var(--text-muted);
      font-family: 'JetBrains Mono', monospace;
    }
    .step-intent {
      font-size: 0.8rem;
      color: var(--text);
      margin-bottom: 4px;
    }
    .step-url {
      font-size: 0.72rem;
      color: var(--text-muted);
      font-family: 'JetBrains Mono', monospace;
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
      <div class="meta-item">
        <span>轮次:</span>
        <span id="turn-count" class="meta-val">0 / 100</span>
      </div>
    </div>
  </header>

  <!-- Main Container -->
  <div class="container">
    <!-- Live Screen View -->
    <div class="screen-area">
      <div class="canvas-wrapper" id="canvas-container">
        <img id="browser-screen" src="data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='1920' height='1080' viewBox='0 0 1920 1080'><rect width='100%' height='100%' fill='%230b0f19'/><text x='50%' y='50%' fill='%23475569' font-size='28' font-family='sans-serif' text-anchor='middle'>等待 CCTV 视频巡检 Agent 启动并同步视频画面...</text></svg>" alt="Browser Stream">
        
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
        <button class="tab-btn" onclick="switchTab('timeline')">🛠️ 动作序列</button>
        <button class="tab-btn" onclick="switchTab('reasoning')">🧠 AI 推理</button>
        <button class="tab-btn" onclick="switchTab('result')">📑 最终报告</button>
      </div>

      <!-- Tab: Video Segments (Default Active) -->
      <div id="tab-segments" class="tab-content">
        <div class="card" style="padding: 10px 14px; background: #182234; border-color: #2563eb;">
          <div style="display: flex; align-items: center; justify-content: space-between; font-size: 0.82rem;">
            <span style="font-weight: 600; color: #93c5fd;">🎯 1分钟视频抽检进行中</span>
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

      <!-- Tab: Action Timeline -->
      <div id="tab-timeline" class="tab-content" style="display: none;">
        <div class="card" style="flex: 1; display: flex; flex-direction: column;">
          <div class="card-title">Browser Use 自动化指令序列</div>
          <div id="timeline-list" style="flex: 1; overflow-y: auto;">
            <div class="empty-placeholder">等待动作记录...</div>
          </div>
        </div>
      </div>

      <!-- Tab: Reasoning -->
      <div id="tab-reasoning" class="tab-content" style="display: none;">
        <div class="card" style="flex: 1;">
          <div class="card-title">Gemini 多模态视觉思考过程 (Reasoning)</div>
          <div id="reasoning-display" class="card-body" style="color: #9cdcfe; font-family: 'JetBrains Mono', monospace; font-size: 0.82rem;">
            等待模型推理...
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
    const turnCount = document.getElementById('turn-count');
    const hudActionName = document.getElementById('hud-action-name');
    const hudActionDesc = document.getElementById('hud-action-desc');
    const promptDisplay = document.getElementById('prompt-display');
    const reasoningDisplay = document.getElementById('reasoning-display');
    const resultDisplay = document.getElementById('result-display');
    const timelineList = document.getElementById('timeline-list');
    const segmentList = document.getElementById('segment-list');
    const violationList = document.getElementById('violation-list');
    const segmentCountEl = document.getElementById('segment-count');
    const violationCountEl = document.getElementById('violation-count');
    const tabSegCountEl = document.getElementById('tab-seg-count');
    const tabViolCountEl = document.getElementById('tab-viol-count');
    const btnTabViol = document.getElementById('btn-tab-viol');
    const auditStats = document.getElementById('audit-stats');

    let currentTab = 'segments';
    let segmentsData = [];
    let violationsData = [];

    function switchTab(tabName) {
      currentTab = tabName;
      const tabNames = ['segments', 'violations', 'timeline', 'reasoning', 'result'];
      document.querySelectorAll('tab-btn').forEach((btn, i) => {
        btn.classList.toggle('active', tabNames[i] === tabName);
      });
      tabNames.forEach(name => {
        const el = document.getElementById(`tab-${name}`);
        if (el) el.style.display = name === tabName ? 'flex' : 'none';
      });
    }

    function connectWS() {
      const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${proto}//${window.location.host}/ws`;
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
      }
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
      if (s.current_url) urlText.textContent = s.current_url;
      turnCount.textContent = `${s.turn || 0} / ${s.max_turns || 100}`;

      if (s.last_action) {
        hudActionName.textContent = s.last_action;
        hudActionDesc.textContent = (s.last_action_args && s.last_action_args.intent) || (s.current_reasoning && s.current_reasoning.slice(0, 100)) || '正在执行抽检动作';
      }

      if (s.current_reasoning) {
        reasoningDisplay.textContent = s.current_reasoning;
      }
      if (s.final_result) {
        resultDisplay.textContent = s.final_result;
      }

      if (s.timeline && s.timeline.length > 0) {
        renderTimeline(s.timeline);
      }

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
        `;
        segmentList.appendChild(div);
      });
      const tabEl = document.getElementById('tab-segments');
      if (tabEl) tabEl.scrollTop = tabEl.scrollHeight;
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
            视觉证据时间点：${v.time_range} • 现场判定：不符合标准
          </div>
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

    function renderTimeline(timeline) {
      timelineList.innerHTML = '';
      timeline.forEach(item => {
        const div = document.createElement('div');
        div.className = 'step-item';
        div.innerHTML = `
          <div class="step-dot"></div>
          <div class="step-content">
            <div class="step-header">
              <span class="step-action-name">${item.action}</span>
              <span class="step-time">[Turn ${item.turn}] ${item.time}</span>
            </div>
            ${item.intent ? `<div class="step-intent">${item.intent}</div>` : ''}
            ${item.url ? `<div class="step-url">${item.url}</div>` : ''}
          </div>
        `;
        timelineList.appendChild(div);
      });
      timelineList.scrollTop = timelineList.scrollHeight;
    }

    fetch('/api/state').then(r => r.json()).then(s => updateState(s)).catch(() => {});
    connectWS();
  </script>
</body>
</html>
"""


class BrowserMonitorClient:
    """Sends browser use frames, action events, and video segment audit results to the monitor server."""

    def __init__(self):
        self.port = int(os.getenv("MONITOR_PORT", "8080"))
        self._session: Optional[aiohttp.ClientSession] = None
        self._segments: List[Dict[str, Any]] = []

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

    def _fire_and_forget(self, payload: Dict[str, Any]):
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._post_event(payload))
        except RuntimeError:
            pass

    async def _post_event(self, payload: Dict[str, Any]):
        try:
            session = await self._get_session()
            async with session.post(f"http://127.0.0.1:{self.port}/api/event", json=payload):
                pass
        except Exception:
            pass

    def start_session(self, prompt: str, max_turns: int = 100):
        self._segments = []
        self._fire_and_forget({
            "type": "state",
            "data": {
                "status": "RUNNING",
                "prompt": prompt,
                "turn": 0,
                "max_turns": max_turns,
                "current_url": "about:blank",
                "current_reasoning": "正在连接浏览器与视频流，准备抽检...",
                "last_action": "启动浏览器",
                "timeline": [],
                "final_result": "",
                "video_segments": [],
                "sop_violations": [],
            },
        })

    def add_video_segment(
        self,
        time_range: str,
        description: str,
        sop_status: str = "COMPLIANT",
        violation_detail: str = "",
        severity: str = "NORMAL",
    ):
        """Records a video segment audit result and immediately pushes it to the right sidebar."""
        sop_status_norm = (sop_status or "COMPLIANT").upper()
        if "VIOLAT" in sop_status_norm or "违规" in sop_status_norm or "不符合" in sop_status_norm:
            sop_status_norm = "VIOLATION"
        elif "CANNOT" in sop_status_norm or "UNKNOWN" in sop_status_norm or "无法" in sop_status_norm:
            sop_status_norm = "CANNOT_DETERMINE"
        else:
            sop_status_norm = "COMPLIANT"

        seg = {
            "id": len(self._segments) + 1,
            "time_range": time_range.strip(),
            "description": description.strip(),
            "sop_status": sop_status_norm,
            "violation_detail": violation_detail.strip() if violation_detail else "",
            "severity": severity.upper() if severity else ("RED_LINE" if sop_status_norm == "VIOLATION" else "NONE"),
            "time": time.strftime("%H:%M:%S"),
        }
        self._segments.append(seg)
        self._fire_and_forget({
            "type": "segment",
            "segment": seg,
        })
        # Direct local checkpoint append for zero-loss 24h auditing
        try:
            ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            with open(os.path.join(ckpt_dir, "audit_records.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(seg, ensure_ascii=False) + "\n")
        except Exception:
            pass
        print(f"📹 [Monitor Right Sidebar] New Segment Added: [{time_range}] Status: {sop_status_norm} - {description[:60]}")

    def update_frame_b64(self, b64_str: str):
        self._fire_and_forget({"type": "frame", "frame": b64_str})

    def update_frame_bytes(self, frame_bytes: bytes, url: Optional[str] = None):
        b64_str = base64.b64encode(frame_bytes).decode("ascii")
        self._fire_and_forget({
            "type": "frame",
            "frame": b64_str,
            "url": url,
        })

    def set_turn_info(self, turn: int, reasoning: str, actions: List[Any]):
        action_names = [a.name if hasattr(a, "name") else str(a) for a in actions]
        last_action = ", ".join(action_names) if action_names else "观察视频画面"
        self._fire_and_forget({
            "type": "state",
            "data": {
                "turn": turn,
                "current_reasoning": reasoning,
                "last_action": last_action,
            },
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
            click_target = {
                "x": int(norm_x * 1920 / 1000),
                "y": int(norm_y * 1080 / 1000),
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


monitor = BrowserMonitorClient()
