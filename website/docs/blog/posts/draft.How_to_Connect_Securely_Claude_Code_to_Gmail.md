---
draft: true
title: ""
authors:
  - gpsaggese
date: 2026-
description:
categories:
  - Causal AI
---

TL;DR: Running blind experiments is expensive. Bayesian Optimization predicts
where to look next and saves you millions.

<!-- more -->

Connect Claude Code to Gmail via MCP (Mac Guide)

A quick setup to let Claude Code read, summarize, and send Gmail using MCP (Model
Context Protocol).

This uses the modern MCP marketplace (Smithery) — not manual OAuth or custom servers.

## Create Google OAuth 2.0 & Gmail API Setup

If you want your application to read, search, or send emails on behalf of a user, Google requires you to use **OAuth 2.0 authorization** instead of passwords or app passwords.

This tutorial walks through the complete, real-world setup developers actually need — from an empty Google Cloud account to a working Gmail API integration.

By the end you will have:
- A Google Cloud project
- Gmail API enabled
- OAuth consent screen configured
- OAuth client credentials created
- A working authorization flow

---

## Why OAuth is Required

Google does not allow apps to access Gmail using usernames + passwords.

Instead your app must:
1. Redirect the user to Google
2. Ask permission for specific scopes
3. Receive a temporary authorization code
4. Exchange it for access + refresh tokens
5. Use tokens to call Gmail API

This protects users and allows them to revoke access anytime.

---

## Step 1 — Create a Google Cloud Project

1. Open https://console.cloud.google.com
2. Click the project selector (top left)
3. Click **New Project**
4. Give it a name (example: `gmail-integration`)
5. Click **Create**

You now have an isolated environment where APIs and credentials live.

---

## Step 2 — Enable the Gmail API

1. Navigate to: **APIs & Services → Library**
2. Search for **Gmail API**
3. Open it
4. Click **Enable**

Nothing will work until this step is done — most OAuth errors later come from forgetting this.

---

## Step 3 — Configure the OAuth Consent Screen

Go to:

**APIs & Services → OAuth consent screen**

### Choose User Type

| Option | When to use |
|------|------|
| External | Personal apps, SaaS, testing |
| Internal | Google Workspace company apps |

Most developers choose **External**.

---

### Fill Required Fields

| Field | Value |
|------|------|
| App name | Your app name |
| User support email | Your email |
| Developer contact | Your email |

Save and continue.

---

### Add Gmail Scopes

Click **Add or Remove Scopes** and add only what you need:

| Scope | Purpose |
|------|------|
| `gmail.readonly` | Read emails |
| `gmail.send` | Send emails |
| `gmail.modify` | Read + modify labels |

⚠️ Use the smallest set possible — fewer scopes avoids verification later.

---

### Add Test Users (Important)

Add your Gmail address here.

If you skip this, authentication fails with:

`access_denied`

Save and finish setup.

---

## Step 4 — Create OAuth Client Credentials

Go to:

**APIs & Services → Credentials → Create Credentials → OAuth client ID**

### Choose Application Type

| Type | Use Case |
|------|------|
| Web Application | Backend server |
| Desktop App | Local scripts |
| Mobile | iOS/Android |

Most integrations use **Web Application**.

---

### Configure Redirect URIs

This must EXACTLY match your application callback.

Example for local development:

http://localhost:3333/oauth/callback

Example production:

https://yourdomain.com/oauth/google/callback

Even a trailing slash causes errors.

Click **Create**.

---

## Step 5 — Download Credentials

You will receive:

- Client ID
- Client Secret

Example JSON:

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

Store this securely — never commit to Git.

⸻

Step 6 — Authorization Flow

1. Redirect User to Google

https://accounts.google.com/o/oauth2/auth
 ?client_id=YOUR_CLIENT_ID
 &redirect_uri=YOUR_REDIRECT
 &response_type=code
 &scope=https://www.googleapis.com/auth/gmail.readonly
 &access_type=offline
 &prompt=consent

2. User Approves

Google redirects to:

http://localhost:3333/oauth/callback?code=AUTH_CODE

3. Exchange Code for Tokens

POST request:

https://oauth2.googleapis.com/token

Body:

code=AUTH_CODE
client_id=CLIENT_ID
client_secret=CLIENT_SECRET
redirect_uri=REDIRECT_URI
grant_type=authorization_code

Response includes:
	•	access_token
	•	refresh_token

Save the refresh token permanently.

⸻

Step 7 — Call the Gmail API

Example: list messages

GET https://gmail.googleapis.com/gmail/v1/users/me/messages
Authorization: Bearer ACCESS_TOKEN


⸻

Common Errors and Fixes

Error	Cause
redirect_uri_mismatch	URI mismatch
access_denied	user not in test users
invalid_scope	Gmail API not enabled
app_not_verified	expected during testing


⸻

When You Need Google Verification

You only need verification if:
	•	You exceed 100 users
	•	You use sensitive scopes publicly

Internal tools and personal apps do NOT require approval.

⸻

Final Checklist
	•	Project created
	•	Gmail API enabled
	•	OAuth consent configured
	•	Test users added
	•	OAuth client created
	•	Redirect URI exact
	•	Tokens exchanged
	•	Gmail API request successful

You now have a production-grade Gmail integration using OAuth 2.0.




## Connect to 

⸻

What you get

After setup you can ask Claude:
	•	“Summarize unread emails”
	•	“Draft a reply to the last message”
	•	“Send email to Alex about tomorrow’s meeting”
	•	“Find invoices from last week”

⸻

1) Install the MCP marketplace

Install the Smithery CLI:

npm install -g @smithery/cli

Login (opens browser OAuth):

smithery login

Install the Gmail integration:

smithery install gmail

This step automatically:
	•	installs the Gmail MCP server
	•	connects your Google account
	•	registers the tool for Claude

⸻

2) Prepare Claude Code config folders

Create the Claude Code config directory:

mkdir ~/.config/claude-code

Check your config home (should usually be empty or ~/.config):

echo $XDG_CONFIG_HOME


⸻

3) Inspect the installed MCP server

Smithery writes a server definition file.

Verify it exists:

ls ~/.config/claude-code/mcp_servers.json

Pretty-print it:

cat ~/.config/claude-code/mcp_servers.json | jq


⸻

4) Link MCP to Claude Desktop

Find where npx lives (Claude uses it to run tools):

which npx

Now edit the MCP config if needed:

vi ~/.config/claude-code/mcp_servers.json

Copy it into Claude’s home config:

cp ~/.config/claude-code/mcp_servers.json ~/.claude/mcp_servers.json

You may also want to inspect the main Claude config:

vi ~/.claude.json


⸻

5) Restart Claude

Fully quit and reopen Claude Desktop.

⸻

6) Test it

Try:

summarize my latest email

If connected correctly, Claude will ask permission once, then call Gmail automatically.

⸻

How it works (in one sentence)

Claude → MCP tool → Gmail API → response → Claude

No passwords in prompts
No scraping
No custom backend

⸻

Troubleshooting

Nothing happens when asking about email
	•	Restart Claude Desktop
	•	Confirm ~/.claude/mcp_servers.json exists

Permission errors
	•	Re-run:

smithery login
smithery install gmail



jq not installed

brew install jq


⸻

You now have a local AI email agent powered by Claude + MCP.
