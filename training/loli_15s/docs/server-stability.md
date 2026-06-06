# Server freezes during teacher gen

## What we think is happening

Hard freezes (no Python traceback, SSH dies, need reboot) are usually **kernel-level**, not app bugs.

From `resource_monitor.log` before the last crash:

| Signal | Last reading |
|--------|----------------|
| RAM used | **~89 GiB / 125 GiB** (only **3–7 GiB** free) |
| Swap | **6.9 GiB** in use |
| GPUs | 3× `moss-tts-server` at **~11 + 9.5 + 19.5 GiB VRAM** |
| Workers | 3× Python teacher shards |

Likely cause: **memory pressure** from running **3 openmoss servers** with `OPENMOSS_AUX_CPU=1` (codec/aux on **CPU RAM**) plus 3 inference workers and `/dev/shm` WAV staging. The kernel hits OOM or stalls in the GPU driver; userspace never logs an error.

Secondary risks: GPU driver hang (NVRM Xid), swap thrashing, concurrent `rsync` every 120s.

## Logging (use on every heavy run)

```bash
# Start watchdog (JSONL + kernel OOM/Xid scrape)
LOG_DIR=training/loli_15s_batch3/logs/health \
STAGING=/dev/shm/loli15s_wavs \
WAV_DISK=training/loli_15s_batch3/wavs \
python3 training/moss-realtime/scripts/legacy/watchdog_server_health.py \
  --log-dir "$LOG_DIR" --interval 15

# After crash or run end
python3 training/moss-realtime/scripts/legacy/analyze_health_log.py \
  training/loli_15s_batch3/logs/health/health.jsonl

# Kernel alerts captured during run
cat training/loli_15s_batch3/logs/health/kernel_alerts.log
cat training/loli_15s_batch3/logs/health/alerts.log
```

Teacher gen enables this automatically when `MONITOR=1` (default).

## Prevention defaults (updated)

| Setting | Safer value | Why |
|---------|-------------|-----|
| `NUM_SHARDS` / `GPUS` | **2** not 3–4 | Less parallel RAM/VRAM |
| `OPENMOSS_AUX_CPU` | **0** when multi-GPU | Stops 3× CPU aux model copies |
| `LIGHT_HOST` | **0** | Avoid mass docker stop/start |
| `CLEAR_SWAP` | **0** | Avoid swapoff needing sudo |
| `MIN_AVAIL_GB` | **12+** | Preflight blocks run if RAM tight |
| `HEALTH_MIN_AVAIL_GB` | **8** | Watchdog alerts before freeze |

```bash
# Safer 2-GPU resume
OPENMOSS_AUX_CPU=0 GPUS=0,1 NUM_SHARDS=2 PORTS=8014,8015 \
MIN_AVAIL_GB=12 FRESH_WAVS=0 SKIP_CORPUS=1 \
bash training/loli_15s/scripts/run_loli_batch3_3k.sh
```

## After a freeze

On next boot:

```bash
sudo dmesg -T | egrep -i 'oom|killed process|nvrm|xid|hung|lockup' | tail -50
journalctl -k -b -1 | tail -100   # previous boot kernel log
```
