#!/usr/bin/env bash
# Full KPC4_1 comparison report: per model, across f16 / q4_0 / q8_0 / kpc(kpc4_1 K + q4_1 V),
# capture PPL, KV-cache (K/V split), total memory (peak RSS), prefill + decode tok/s.
set -u
B=${LLAMA_BIN:-./build/bin}
WIKI=${WIKI:-/tmp/wikitext-2-raw/wiki.test.raw}
T=${T:-6}
OUT=${OUT:-/tmp/kpc_report}
mkdir -p "$OUT"

# model | path | ppl-chunks
MODELS=(
  "qwen05b      | /tmp/tbq_models/qwen05b.gguf                                          | 24"
  "llama1b      | /tmp/tbq_models/llama1b.gguf                                          | 24"
  "qwen35-4b    | /tmp/tbq_models/qwen35-4b.gguf                                        | 16"
  "deepseek1.3b | /home/mike/models/deepseek-1.3b/deepseek-coder-1.3b-instruct.Q8_0.gguf | 24"
  "glm4-9b      | /home/mike/models/glm4-9b/glm-4-9b-chat-Q4_K_M.gguf                   | 12"
  "gemma2-2b    | /home/mike/models/gemma2-2b/gemma-2-2b-it-Q8_0.gguf                   | 16"
)
# label | ctk | ctv
CFGS=( "f16|f16|f16" "q4_0|q4_0|q4_0" "q8_0|q8_0|q8_0" "kpc|kpc4_1|q4_1" )

num()  { grep -oE '[0-9]+\.[0-9]+|[0-9]+' | head -1; }

for ms in "${MODELS[@]}"; do
  IFS='|' read -r M MP NCH <<< "$ms"; M=$(echo "$M"|xargs); MP=$(echo "$MP"|xargs); NCH=$(echo "$NCH"|xargs)
  if [ ! -f "$MP" ]; then echo "## $M : SKIP (not found: $MP)"; echo; continue; fi
  msize=$(du -h "$MP" | cut -f1)
  echo "############################################################"
  echo "# MODEL: $M   (file ${msize}, PPL ${NCH} chunks @ ctx512, -fa on, threads ${T})"
  echo "############################################################"
  printf '%-5s | %-9s | %-9s | %-22s | %-9s | %-12s | %-12s\n' \
    cfg PPL KV-MiB "KV split (K / V)" RSS-MiB "pp512 t/s" "tg128 t/s"
  printf -- '------+-----------+-----------+------------------------+-----------+--------------+-------------\n'
  for cfg in "${CFGS[@]}"; do
    IFS='|' read -r name ctk ctv <<< "$cfg"
    rl="$OUT/${M}_${name}_run.log"; bl="$OUT/${M}_${name}_bench.log"

    # one timed run -> PPL + KV-cache + peak RSS  (-v exposes the kv_cache size line)
    /usr/bin/time -v "$B/llama-perplexity" -m "$MP" -f "$WIKI" -c 512 --chunks "$NCH" \
        -fa on -kvu -ctk "$ctk" -ctv "$ctv" -t "$T" -v >/dev/null 2> "$rl"
    ppl=$(grep -oE 'Final estimate: PPL = [0-9.]+' "$rl" | num); : "${ppl:=ERR}"
    kvline=$(grep -m1 'llama_kv_cache: size =' "$rl")
    kvtot=$(echo "$kvline" | sed -nE 's/.*size = *([0-9.]+) MiB.*/\1/p'); : "${kvtot:=?}"
    kvk=$(echo "$kvline" | sed -nE 's/.*K \([^)]*\): *([0-9.]+) MiB.*/\1/p'); : "${kvk:=?}"
    kvv=$(echo "$kvline" | sed -nE 's/.*V \([^)]*\): *([0-9.]+) MiB.*/\1/p'); : "${kvv:=?}"
    rss_kb=$(grep -i 'maximum resident' "$rl" | num); : "${rss_kb:=0}"
    rss=$(awk -v k="$rss_kb" 'BEGIN{printf (k>0)?"%.0f":"?", k/1024}')

    # throughput: prefill (pp512) + decode (tg128). No -d: KPC refuses the context-shift that
    # llama-bench's -d triggers, and t/s is the last "| <val> ± <val> |" column (not the size column).
    timeout 900 "$B/llama-bench" -m "$MP" -p 512 -n 128 -fa 1 \
        -ctk "$ctk" -ctv "$ctv" -t "$T" -r 2 > "$bl" 2>&1
    pp=$(grep -E 'pp512' "$bl" | sed -nE 's/.*\| +([0-9.]+) ± [0-9.]+ \|.*/\1/p' | head -1); : "${pp:=ERR}"
    tg=$(grep -E 'tg128' "$bl" | sed -nE 's/.*\| +([0-9.]+) ± [0-9.]+ \|.*/\1/p' | head -1); : "${tg:=ERR}"

    printf '%-5s | %-9s | %-9s | %-22s | %-9s | %-12s | %-12s\n' \
      "$name" "$ppl" "$kvtot" "${kvk} / ${kvv}" "$rss" "$pp" "$tg"
  done
  echo
done
echo REPORT_DONE
