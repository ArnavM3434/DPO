#!/bin/bash
# Watch checkpoints/, upload finished ones to Hugging Face, delete after success.
#
# Run in tmux on a login node (needs network + hf auth):
#   hf auth login
#   tmux new -s sft-sync
#   bash scripts/sync_sft_checkpoints.sh
#
# Env overrides:
#   CHECKPOINTS_DIR   default: $PROJECT_DIR/checkpoints
#   HF_REPO_ID        default: ArnavM3434/sft-try-again
#   INTERVAL_SEC      default: 300  (scan every 5 min)
#   STABLE_SEC        default: 120  (wait 2 min after last file change)

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/work/hdd/bfgp/arnavm7/DPO}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${PROJECT_DIR}/checkpoints}"
HF_REPO_ID="${HF_REPO_ID:-ArnavM3434/sft-try-again}"
INTERVAL_SEC="${INTERVAL_SEC:-300}"
STABLE_SEC="${STABLE_SEC:-120}"
LOG_FILE="${LOG_FILE:-${PROJECT_DIR}/logs/checkpoint-sync.log}"

mkdir -p "$(dirname "$LOG_FILE")"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

checkpoint_is_stable() {
  local dir="$1"
  # Upload only if nothing in the folder changed recently (trainer likely done writing).
  if find "$dir" -type f -mmin "-$((STABLE_SEC / 60))" -print -quit | grep -q .; then
    return 1
  fi
  [[ -f "$dir/adapter_config.json" && -f "$dir/adapter_model.safetensors" ]]
}

upload_checkpoint() {
  local dir="$1"
  local name
  name="$(basename "$dir")"

  log "Uploading ${dir} -> ${HF_REPO_ID}/${name}/"
  hf upload "$HF_REPO_ID" "$dir" "${name}/" \
    --exclude "optimizer.pt" \
    --exclude "scheduler.pt" \
    --exclude "rng_state.pth" \
    --exclude "trainer_state.json" \
    --exclude "training_args.bin"
}

remove_checkpoint() {
  local dir="$1"
  log "Removing local ${dir}"
  rm -rf "$dir"
}

scan_once() {
  shopt -s nullglob
  local dirs=("$CHECKPOINTS_DIR"/checkpoint-*)
  shopt -u nullglob

  if ((${#dirs[@]} == 0)); then
    log "No checkpoints in ${CHECKPOINTS_DIR}"
    return 0
  fi

  for dir in "${dirs[@]}"; do
    [[ -d "$dir" ]] || continue

    if ! checkpoint_is_stable "$dir"; then
      log "Skipping ${dir} (missing adapter files or still being written)"
      continue
    fi

    if upload_checkpoint "$dir"; then
      remove_checkpoint "$dir"
      log "Done: $(basename "$dir")"
    else
      log "Upload failed for ${dir}; keeping local copy"
    fi
  done
}

if ! command -v hf >/dev/null 2>&1; then
  echo "hf CLI not found. Install huggingface_hub and run: hf auth login" >&2
  exit 1
fi

log "Starting checkpoint sync"
log "  dir:      ${CHECKPOINTS_DIR}"
log "  repo:     ${HF_REPO_ID}/checkpoint-N/"
log "  interval: ${INTERVAL_SEC}s"
log "  stable:   ${STABLE_SEC}s"

while true; do
  scan_once || log "Scan error (continuing)"
  sleep "$INTERVAL_SEC"
done
