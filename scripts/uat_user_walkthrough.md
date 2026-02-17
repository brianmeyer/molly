# Molly UAT Walkthrough

Manual acceptance tests for verifying Molly works end-to-end.
Run **after** `scripts/uat_full.py` passes all automated checks.

---

## Prerequisites

```bash
# 1. Automated UAT passes
python scripts/uat_full.py --no-e2e

# 2. Docker running, Neo4j healthy
docker ps | grep neo4j   # should show "Up"

# 3. WhatsApp auth exists
ls store/auth/            # should have session files
```

---

## Test 1: Cold Start

```bash
# Start Molly fresh
python main.py
```

**Expected log output (in order):**
- [ ] `Running preflight checks...`
- [ ] `Docker daemon: running`
- [ ] `Neo4j container: already running`
- [ ] `Neo4j bolt port: ready`
- [ ] `Triage model ready:`
- [ ] `Google OAuth: authenticated`
- [ ] `Prewarming ML models...`
- [ ] `Embedding model prewarmed`
- [ ] `GLiNER2 model prewarmed`
- [ ] `Health preflight: completed`
- [ ] `Preflight checks passed`
- [ ] `WhatsApp connected`

**Time budget:** < 60s from start to "WhatsApp connected"

---

## Test 2: Basic Chat (Owner DM)

Send via WhatsApp DM to Molly:
> "What's 2+2?"

- [ ] Response arrives within 15s
- [ ] Response contains "4"
- [ ] No errors in terminal output

---

## Test 3: Memory Pipeline

Send via WhatsApp:
> "Remember that I'm meeting Jake at Blue Bottle Coffee tomorrow at 3pm"

- [ ] Molly acknowledges the message
- [ ] Wait 5s, then send: "What am I doing tomorrow?"
- [ ] Response mentions Jake, Blue Bottle, or 3pm

---

## Test 4: Commands

Send each command via WhatsApp DM:

| Command | Expected |
|---------|----------|
| `/help` | Lists available commands |
| `/memory search coffee` | Returns memory results (or "no results") |
| `/graph Brian` | Shows entity connections (or "no entities found") |

- [ ] All 3 commands respond without error

---

## Test 5: Group Chat Modes

In a registered WhatsApp group:
1. Send a message **without** @Molly tag
   - [ ] Molly does NOT respond (passive processing only)
2. Send a message **with** @Molly tag
   - [ ] Molly responds normally

---

## Test 6: Approval Flow

Send via WhatsApp:
> "Send a message to Mom saying I'll be late"

- [ ] Molly asks for confirmation (CONFIRM tier)
- [ ] Reply "yes" → action executes
- [ ] Reply "no" → action cancelled

---

## Test 7: Echo Loop Prevention

After Molly sends a response:
- [ ] Molly does NOT respond to its own messages
- [ ] No infinite loops in terminal output

---

## Test 8: Graceful Degradation

### 8a. Neo4j down
```bash
docker stop neo4j
```
Send a WhatsApp message:
- [ ] Molly still responds (degraded, no graph context)
- [ ] Log shows: `Neo4j unavailable — graph layer disabled`
- [ ] No crash

```bash
docker start neo4j   # restore
```

### 8b. No internet (optional)
Disconnect WiFi briefly, send a message:
- [ ] Molly fails gracefully (timeout error, not crash)
- [ ] Reconnect → next message works normally

---

## Test 9: Health Report

Wait for nightly maintenance (or trigger manually):
```bash
# In a separate terminal, while Molly is running:
python -c "
from monitoring.health import HealthDoctor
d = HealthDoctor()
report = d.run_full_check()
print(report.summary if hasattr(report, 'summary') else report)
"
```

- [ ] Report generates without crash
- [ ] Shows check results for multiple layers

---

## Test 10: Restart

```bash
# Send SIGHUP to trigger graceful restart
kill -HUP $(cat store/.molly.instance.lock)
```

- [ ] Molly logs: `Restart requested: SIGHUP received`
- [ ] Process exits with code 42
- [ ] `run_molly.sh` restarts Molly automatically
- [ ] WhatsApp reconnects

---

## Test 11: Clean Shutdown

Press `Ctrl+C` in the terminal:
- [ ] Molly logs: `Interrupted.`
- [ ] Process exits cleanly (code 130)
- [ ] No zombie processes left (`ps aux | grep main.py`)

---

## Results

| Test | Pass/Fail | Notes |
|------|-----------|-------|
| 1. Cold Start | | |
| 2. Basic Chat | | |
| 3. Memory Pipeline | | |
| 4. Commands | | |
| 5. Group Chat Modes | | |
| 6. Approval Flow | | |
| 7. Echo Loop | | |
| 8a. Neo4j Down | | |
| 8b. No Internet | | |
| 9. Health Report | | |
| 10. Restart | | |
| 11. Clean Shutdown | | |

**Date:** ___________  **Tester:** ___________
