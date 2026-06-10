#!/usr/bin/env bash
# Decode/prefill throughput across f16/q4_0/q8_0/kpc. Plain pp512 + tg128 from a fresh cache,
# comparable across all configs (no -d so every config measures the same fresh-cache work).
set -u
B=${LLAMA_BIN:-./build/bin}
T=${T:-6}
OUT=${OUT:-/tmp/kpc_report}
mkdir -p "$OUT"

MODELS=(
  "qwen05b   | /tmp/tbq_models/qwen05b.gguf"
  "llama1b   | /tmp/tbq_models/llama1b.gguf"
  "qwen35-4b | /tmp/tbq_models/qwen35-4b.gguf"
)
CFGS=( "f16|f16|f16" "q4_0|q4_0|q4_0" "q8_0|q8_0|q8_0" "kpc|kpc4_1|q4_1" )

# t/s is the last "| <float> ± <float> |" column
ts() { grep -E "$1" "$2" | sed -nE 's/.*\| +([0-9.]+) ± [0-9.]+ \|.*/\1/p' | head -1; }

for ms in "${MODELS[@]}"; do
  IFS='|' read -r M MP <<< "$ms"; M=$(echo "$M"|xargs); MP=$(echo "$MP"|xargs)
  [ -f "$MP" ] || { echo "$M : SKIP"; continue; }
  printf '%-9s | %-5s | %-12s | %-12s\n' "$M" cfg "pp512 t/s" "tg128 t/s"
  for cfg in "${CFGS[@]}"; do
    IFS='|' read -r name ctk ctv <<< "$cfg"
    bl="$OUT/${M}_${name}_perf.log"
    timeout 600 "$B/llama-bench" -m "$MP" -p 512 -n 128 -fa 1 \
        -ctk "$ctk" -ctv "$ctv" -t "$T" -r 3 > "$bl" 2>&1
    pp=$(ts 'pp512' "$bl"); : "${pp:=ERR}"
    tg=$(ts 'tg128' "$bl"); : "${tg:=ERR}"
    printf '%-9s | %-5s | %-12s | %-12s\n' "" "$name" "$pp" "$tg"
  done
  echo
done
echo PERF_DONE
