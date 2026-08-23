# Daily signal notifier

A tiny GitHub Actions job that runs once a day, computes the **SPY** and **QQQ**
vote-of-2 signals plus six standalone **SMA-200** checks, and pushes a summary to
**Telegram**. It reuses the LETF Lab engine (`ai_swing`) so the numbers match the app.

- `watchlist.py` — what gets tracked (edit here to change it).
- `daily_signals.py` — computes signals, diffs vs the last run, sends Telegram.
- `state/last_signals.json` — previous run's booleans; the workflow commits it back
  each day so "Changed today" works. Do not edit by hand (except to test).
- `../.github/workflows/daily-signals.yml` — the schedule + job.

## What the message looks like

```
📊 LETF Lab — 2026-08-21

SPY  🟢 RISK-ON (3/4)
   SMA250 ✓  SMA100 ✓  Vol21d ✓  AR(1) ✗
QQQ  🟢 RISK-ON (4/4)
   SMA250 ✓  SMA100 ✓  Vol21d ✓  AR(1) ✓

SMA-200 watch
  ✗  SPY < 200SMA
  ✓  SPY > 200SMA +4%
  ✗  SPY < 200SMA −3%
  ✗  QQQ < 200SMA
  ✓  QQQ > 200SMA +4%
  ✗  QQQ < 200SMA −3%

⚠️ Changed today
  SPY > 200SMA +4%: cleared → triggered
```

## One-time setup

### 1. Create a Telegram bot and get your chat ID
1. In Telegram, message **@BotFather** → `/newbot` → follow prompts → copy the **bot token**.
2. Send any message to your new bot (so it can reply to you).
3. Get your **chat ID**: open
   `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` in a browser and read
   `result[0].message.chat.id` (a number, sometimes negative for groups).

### 2. Add repo secrets
In the fork on GitHub → **Settings → Secrets and variables → Actions → New repository secret**:
- `TELEGRAM_BOT_TOKEN` — the BotFather token.
- `TELEGRAM_CHAT_ID` — the chat ID.

### 3. Test it
- GitHub → **Actions → Daily LETF signals → Run workflow** (this is `workflow_dispatch`).
- Confirm the message arrives on your phone.

## Local dry run (no secrets, no send)

From the repo root, with the backend installed (`pip install ./backend`):

```bash
PRICE_CACHE_DIR=./data/prices python -m notify.daily_signals --dry-run
```

Prints the message to stdout without sending to Telegram or saving state.

## Schedule / timezone

Cron runs at `0 22 * * 1-5` (22:00 UTC, weekdays) — after the US close year-round.
GitHub cron has no timezone and does not observe DST, so the ET time shifts by an
hour twice a year (18:00 ET in summer, 17:00 ET in winter). Edit the cron in the
workflow if you want a different time. Note GitHub's scheduled runs can be delayed
several minutes under load.

## Changing what's tracked

Edit `watchlist.py`:
- `STRATEGIES` — add/remove vote-of-k signals (any benchmark ticker, any of the
  engine's indicator types: `SMA_GATE`, `EMA_GATE`, `VOL_GATE`, `AR1_GATE`).
- `SMA200_CHECKS` — add/remove standalone price-vs-SMA checks (`op` is `lt`/`gt`,
  `factor` multiplies the SMA, e.g. `1.05` for +5%).

No other file needs to change.
