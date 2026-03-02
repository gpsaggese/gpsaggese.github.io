---
draft: true
title: ""
authors:
  - gpsaggese
date: 2026-03-02
description:
categories:
  - Causal AI
---

TL;DR:

<!-- more -->

// From https://code.claude.com/docs/en/hooks-guide

{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "jq -r '.tool_input.file_path' | xargs npx prettier --write"
          }
        ]
      }
    ]
  }
}
