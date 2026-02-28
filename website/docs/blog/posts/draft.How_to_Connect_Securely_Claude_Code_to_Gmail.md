---
draft: true
title: "How to Connect Claude Code Securely to Gmail"
authors:
  - gpsaggese
date: 2026-02-28
description:
categories:
  - Causal AI
---

TL;DR: Connect Claude Code to Gmail in minutes using MCP — no passwords in
prompts, no custom backend, just clean OAuth and Smithery.

<!-- more -->

- This tutorial shows you how to connect Claude Code to Gmail using MCP (Model
  Context Protocol), so Claude can read, summarize, and send emails on your
  behalf. We use the modern MCP marketplace via Smithery — no manual OAuth
  servers, no credential juggling.
- By the end, you will have Claude Code reading and composing Gmail directly
  from the command line.

## Part 1: Google OAuth 2.0 and Gmail API Setup

- If you want your application to read, search, or send emails on behalf of a
  user, Google requires OAuth 2.0 authorization instead of passwords or app
  passwords.
- This section walks through the complete setup — from an empty Google Cloud
  account to a working Gmail API integration.
- By the end you will have:
  - A Google Cloud project
  - Gmail API enabled
  - OAuth consent screen configured
  - OAuth client credentials created
  - A working authorization flow

## Why OAuth Is Required

- Google does not allow apps to access Gmail using usernames and passwords.
- Instead, your app must:
  - Redirect the user to Google
  - Ask permission for specific scopes
  - Receive a temporary authorization code
  - Exchange it for access and refresh tokens
  - Use tokens to call the Gmail API
- This protects users and allows them to revoke access at any time.

## Step 1 — Create a Google Cloud Project

- Open [Google Cloud Console](https://console.cloud.google.com)
- Click the project selector (top left)
- Click **New Project**
- Give it a name (example: `gmail-integration`)
- Click **Create**

- You now have an isolated environment where APIs and credentials live.

## Step 2 — Enable the Gmail API

- Navigate to **APIs & Services → Library**
- Search for **Gmail API**
- Open it
- Click **Enable**

- Most OAuth errors come from forgetting this step.

## Step 3 — Configure the OAuth Consent Screen

- Go to **APIs & Services → OAuth consent screen**.

### Choose User Type

| Option | When to Use |
| :----- | :---------- |
| External | Personal apps, SaaS, testing |
| Internal | Google Workspace company apps |

- Most developers choose **External**.

### Fill Required Fields

| Field | Value |
| :---- | :---- |
| App name | Your app name |
| User support email | Your email |
| Developer contact | Your email |

- Save and continue.

### Add Gmail Scopes

- Click **Add or Remove Scopes** and add only what you need:

| Scope | Purpose |
| :---- | :------ |
| `gmail.readonly` | Read emails |
| `gmail.send` | Send emails |
| `gmail.modify` | Read and modify labels |

- Use the smallest set possible — fewer scopes avoids verification later.

### Add Test Users

- Add your Gmail address here. If you skip this, authentication fails with
  `access_denied`.
- Save and finish setup.

## Step 4 — Create OAuth Client Credentials

- Go to **APIs & Services → Credentials → Create Credentials → OAuth client
  ID**.

### Choose Application Type

| Type | Use Case |
| :--- | :------- |
| Web Application | Backend server |
| Desktop App | Local scripts |
| Mobile | iOS/Android |

- Most integrations use **Web Application**.

### Configure Redirect URIs

- The redirect URI must exactly match your application callback — even a
  trailing slash causes errors.
- Example for local development:
  ```text
  http://localhost:3333/oauth/callback
  ```
- Example for production:
  ```text
  https://yourdomain.com/oauth/google/callback
  ```
- Click **Create**.

## Step 5 — Download Credentials

- You will receive a Client ID and Client Secret. Download the JSON file:
  ```json
  {
    "web": {
      "client_id": "xxxx.apps.googleusercontent.com",
      "client_secret": "XXXX",
      "auth_uri": "https://accounts.google.com/o/oauth2/auth",
      "token_uri": "https://oauth2.googleapis.com/token",
      "redirect_uris": ["http://localhost:3333/oauth/callback"]
    }
  }
  ```
- Store this securely — never commit it to Git.

## Step 6 — Authorization Flow

- Redirect the user to Google:
  ```text
  https://accounts.google.com/o/oauth2/auth
    ?client_id=YOUR_CLIENT_ID
    &redirect_uri=YOUR_REDIRECT
    &response_type=code
    &scope=https://www.googleapis.com/auth/gmail.readonly
    &access_type=offline
    &prompt=consent
  ```
- User approves — Google redirects to:
  ```text
  http://localhost:3333/oauth/callback?code=AUTH_CODE
  ```
- Exchange the code for tokens via POST to
  `https://oauth2.googleapis.com/token`:
  ```text
  code=AUTH_CODE
  client_id=CLIENT_ID
  client_secret=CLIENT_SECRET
  redirect_uri=REDIRECT_URI
  grant_type=authorization_code
  ```
- The response includes an `access_token` and a `refresh_token`. Save the
  refresh token permanently.

## Step 7 — Call the Gmail API

- Example — list messages:
  ```text
  GET https://gmail.googleapis.com/gmail/v1/users/me/messages
  Authorization: Bearer ACCESS_TOKEN
  ```

## Common Errors and Fixes

| Error | Cause |
| :---- | :---- |
| `redirect_uri_mismatch` | URI does not exactly match |
| `access_denied` | User not added to test users |
| `invalid_scope` | Gmail API not enabled |
| `app_not_verified` | Expected during testing phase |

## When You Need Google Verification

- You only need verification if:
  - You exceed 100 users
  - You use sensitive scopes publicly
- Internal tools and personal apps do not require approval.

## Final Checklist

- Project created
- Gmail API enabled
- OAuth consent configured
- Test users added
- OAuth client created
- Redirect URI exact match
- Tokens exchanged
- Gmail API request successful

- You now have a production-grade Gmail integration using OAuth 2.0.

## Part 2: Connect Claude Code to Gmail via MCP

- After completing the OAuth setup, you can connect Claude Code to Gmail using
  the Smithery MCP marketplace.
- After setup you can ask Claude:
  - "Summarize unread emails"
  - "Draft a reply to the last message"
  - "Send email to Alex about tomorrow's meeting"
  - "Find invoices from last week"

### Step 1 — Install the MCP Marketplace

- Install the Smithery CLI:
  ```bash
  > npm install -g @smithery/cli
  ```
- Login (opens browser OAuth):
  ```bash
  > smithery login
  ```
- Install the Gmail integration:
  ```bash
  > smithery install gmail
  ```
- This step automatically:
  - Installs the Gmail MCP server
  - Connects your Google account
  - Registers the tool for Claude

### Step 2 — Prepare Claude Code Config Folders

- Create the Claude Code config directory:
  ```bash
  > mkdir ~/.config/claude-code
  ```
- Check your config home:
  ```bash
  > echo $XDG_CONFIG_HOME
  ```

### Step 3 — Inspect the Installed MCP Server

- Smithery writes a server definition file. Verify it exists:
  ```bash
  > ls ~/.config/claude-code/mcp_servers.json
  ```
- Pretty-print it:
  ```bash
  > cat ~/.config/claude-code/mcp_servers.json | jq
  ```

### Step 4 — Link MCP to Claude Desktop

- Find where `npx` lives:
  ```bash
  > which npx
  ```
- Edit the MCP config if needed:
  ```bash
  > vi ~/.config/claude-code/mcp_servers.json
  ```
- Copy it into Claude's home config:
  ```bash
  > cp ~/.config/claude-code/mcp_servers.json ~/.claude/mcp_servers.json
  ```
- You may also want to inspect the main Claude config:
  ```bash
  > vi ~/.claude.json
  ```

### Step 5 — Restart Claude

- Fully quit and reopen Claude Desktop.

### Step 6 — Test It

- Try:
  ```claude
  claude> summarize my latest email
  ```
- If connected correctly, Claude will ask permission once, then call Gmail
  automatically.

## How It Works

- The full flow is simple:
  ```text
  Claude → MCP tool → Gmail API → response → Claude
  ```
- No passwords in prompts
- No scraping
- No custom backend

## Troubleshooting

- Nothing happens when asking about email:
  - Restart Claude Desktop
  - Confirm `~/.claude/mcp_servers.json` exists
- Permission errors — re-run:
  ```bash
  > smithery login
  > smithery install gmail
  ```
- `jq` not installed:
  ```bash
  > brew install jq
  ```
- You now have a local AI email agent powered by Claude Code and MCP.
