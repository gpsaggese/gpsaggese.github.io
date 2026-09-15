---
title: "TIL: Running Jupyter Notebooks with Apple's Container Engine"
draft: true
authors:
  - gpsaggese
date: 2026-06-13
description: >
  Apple's native container tool (container run) has a bug where -p port
  forwarding accepts TCP connections but resets the return path. This post
  explains the root cause, the debugging journey, and the fix: a standalone
  Python TCP forwarder script that relays traffic through the working bridge100
  interface.
categories:
  - Software Engineering
  - Developer Tools
---

// TODO(ai_gp): Use the link to the GitHub repo dir instead of class_project/project_template/

TL;DR: Apple's `container run -p` port forwarding is broken in v1.0.0 (accepts
TCP but resets on response). The host can reach containers via the vmnet
bridge100 interface though. The fix is to run a separate script that
extracts the container's 192.168.64.x IP and forwards `localhost:PORT` ↔
`container_IP:PORT`.

<!-- more -->

## The Problem

- I was running Jupyter Lab inside an Apple container on macOS:

  ```bash
  container run -p 8888:8888 ... gpsaggese/umd_project_l12_reinforcement_learning
  ```

- Jupyter started fine inside the container (listening on `0.0.0.0:8888`), and
  the `container` process bound port 8888 on the host
- But my browser couldn't connect and the connection was being reset

## Debugging

### Step 1: Check the container internals

I verified Jupyter was actually running and listening:

```bash
$ container exec <name> cat /proc/net/tcp
  sl  local_address rem_address   st ...
   0: 00000000:22B8 00000000:0000 0A ...
```

`0.0.0.0:22B8` = `0.0.0.0:8888`, and `0A` = `TCP_LISTEN`. Jupyter was fine.

### Step 2: Test port forwarding

```bash
$ curl -v http://localhost:8888/
* Connected to localhost (127.0.0.1) port 8888
> GET / HTTP/1.1
> ...
* Recv failure: Connection reset by peer
```

The TCP handshake succeeded (SYN → SYN-ACK → ACK), the HTTP request was sent,
but the response never came back — **"Connection reset by peer"**. The NAT
return path in `container-network-vmnet` was broken.

### Step 3: Discover the bridge100 interface

I checked the host's routing table and found a vmnet bridge:

```bash
$ netstat -rn | grep 192.168
192.168.64         link#20            UC              bridge100
192.168.64.38      f2.c0.f6.57.c2.d0  UHLWIi          bridge100   1199
```

The container had IP `192.168.64.38` on the `bridge100` interface (macOS's
vmnet NAT bridge). The host's bridge IP was `192.168.64.1`.

### Step 4: Test direct bridge connection

```bash
$ curl -4 http://192.168.64.38:8888/
HTTP/1.1 302 Found  # Jupyter redirecting to /lab
```

**It worked.** The bridge100 interface correctly routed traffic to the
container. The bug was specifically in the `-p` port forwarding NAT code, not
the underlying network.

## Root Cause

Apple's `container-network-vmnet` plugin in NAT mode (version 1.0.0):

- Correctly accepts inbound TCP connections on the host side
- Forwards them to the container's VM
- **But fails to relay the return traffic** — the response path through the NAT
  returns a TCP RST

This is a bug in Apple's early-stage container runtime

The `container ps` and `container attach` plugins aren't even installed by
default with the Homebrew package, which confirms the tooling is still maturing.

## The Fix: External Port Forwarding

Since the bridge100 interface works, we can bypass the broken `-p` and forward
ports at the application level.

### The approach

1. `docker_jupyter.sh` starts the container _without_ `-p` (Apple engine
   detection skips it automatically)
2. `docker_jupyter.sh` prints the container's bridge100 IP and a command to run
3. The user opens a second terminal and runs
   `docker_jupyter_port_forward.py`, a standalone script that:
   - Gets the container's bridge100 IP
   - Starts a Python TCP forwarder relaying
     `localhost:<PORT>` → `<container_IP>:<PORT>`
   - Runs in the foreground until Ctrl+C

### Usage

- Start the container on Terminal 1: 
  ```bash
  > DOCKER_ENGINE=apple ./docker_jupyter.sh -f

  Apple container engine detected.
  NOTE: Apple's container tool has a bug where -p port forwarding does
  not work. To access Jupyter from your browser, run this in another
  terminal after the container starts:

    ./docker_jupyter_port_forward.py umd_project_l12_reinforcement_learning.jupyter 8888

  Container IP: 192.168.64.54
  Direct URL: http://192.168.64.54:8888
  ...
  [I 2026-06-13 Jupyter Server 2.19.0 is running at:
  http://127.0.0.1:8888/lab
  ```

- On terminal 2 Set up port forwarding
  ```
  > docker_jupyter_port_forward.py umd_project_l12_reinforcement_learning.jupyter 8888

  Container: umd_project_l12_reinforcement_learning.jupyter
  Bridge IP: 192.168.64.54
  Forwarding localhost:8888 -> 192.168.64.54:8888
  Open http://localhost:8888 in your browser
  Press Ctrl+C to stop.
  ```

- Then open `http://localhost:8888` in your browser

- The forwarder script is at `dev_scripts_helpers/docker/docker_jupyter_port_forward.py`

## References

- The utils live in
  [`class_project/project_template/utils.sh`](/class_project/project_template/utils.sh)
  (functions: `get_container_ip`, `get_docker_jupyter_options`)
- The integration is in
  [`msml610/tutorials/L12_reinforcement_learning/docker_jupyter.sh`](/msml610/tutorials/L12_reinforcement_learning/docker_jupyter.sh)
- The port forward script:
  [`docker_jupyter_port_forward.py`](/msml610/tutorials/L12_reinforcement_learning/docker_jupyter_port_forward.py)
- Apple's container CLI docs: `container --help`
