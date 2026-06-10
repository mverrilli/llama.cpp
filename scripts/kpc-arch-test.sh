#!/usr/bin/env bash
# KPC4_1 architecture coverage on real models: softcap, sinks, ALiBi, SWA cell recycling,
# and the -fa off path; prints a PPL-vs-f16 + KV-cache + tok/s table per architecture.
# Edit MODELS to point at GGUFs you have; missing ones are skipped. Use gemma-2, not gemma-3.
set -u

B=${LLAMA_BIN:-./build/bin}
WIKI=${WIKI:-/tmp/wikitext-2-raw/wiki.test.raw}
T=${T:-$(nproc)}
CH=${CHUNKS:-24}

# name | model path (EDIT to your GGUFs) | feature exercised | swa(1/0) | swa window
# window is used only by SWA_OVERFLOW=1; set it >= the model's real n_swa or recycling won't trigger
MODELS=(
  "gemma2-2b   | $HOME/models/gemma-2-2b-it-Q8_0.gguf           | softcap+SWA+GQA | 1 | 4096"
  "bloom-560m  | $HOME/models/bloom-560m-f16.gguf               | ALiBi+MHA       | 0 | 0"
  "gpt-oss-20b | $HOME/models/gpt-oss-20b-mxfp4.gguf            | sinks+SWA       | 1 | 128"
  "phi3-mini   | $HOME/models/Phi-3-mini-4k-instruct-fp16.gguf  | SWA+GQA         | 1 | 2048"
  "qwen05b     | /tmp/tbq_models/qwen05b.gguf                   | GQA baseline    | 0 | 0"
)

ppl_of() { grep -oE 'Final estimate: PPL = [0-9.]+' "$1" | grep -oE '[0-9.]+$' | head -1; }
kv_of()  { grep -iE 'KV buffer size|KV self size|KV cache size' "$1" | grep -oE '[0-9]+\.[0-9]+ MiB' | head -1; }

hdr() { printf '%-12s | %-16s | %-9s | %-9s | %-8s | %-9s | %-10s | %-8s | %-6s\n' "$@"; }
hdr arch feature KPC-PPL f16-PPL dPPL faoff KV-cache tg-t/s decode
printf -- '-------------+------------------+-----------+-----------+----------+-----------+------------+----------+-------\n'

for spec in "${MODELS[@]}"; do
  IFS='|' read -r name mp feat swa win <<< "$spec"
  name=$(echo "$name" | xargs); mp=$(echo "$mp" | xargs); feat=$(echo "$feat" | xargs)
  swa=$(echo "$swa" | xargs); win=$(echo "$win" | xargs)

  if [ ! -f "$mp" ]; then
    printf '%-12s | %-16s | SKIP (not found: %s)\n' "$name" "$feat" "$mp"
    continue
  fi
  base=/tmp/kpcarch_${name}

  # fused KPC path (the primary one) + KV-cache size
  timeout 1800 "$B/llama-perplexity" -m "$mp" -f "$WIKI" -c 512 --chunks "$CH" \
      -fa on -kvu -ctk kpc4_1 -ctv q4_1 -t "$T" > "${base}_kpc.log" 2>&1
  kpc=$(ppl_of "${base}_kpc.log"); : "${kpc:=ERR}"
  kv=$(kv_of "${base}_kpc.log");   : "${kv:=?}"

  # f16-KV baseline
  timeout 1800 "$B/llama-perplexity" -m "$mp" -f "$WIKI" -c 512 --chunks "$CH" \
      -fa on -kvu -ctk f16 -ctv f16 -t "$T" > "${base}_f16.log" 2>&1
  f16=$(ppl_of "${base}_f16.log"); : "${f16:=ERR}"

  d="?"
  if [[ "$kpc" =~ ^[0-9.]+$ && "$f16" =~ ^[0-9.]+$ ]]; then
    d=$(awk -v a="$kpc" -v b="$f16" 'BEGIN{printf "%+.3f", a-b}')
  fi

  # -fa off dequant path (quantized V needs FA for every type, so use f16 V here)
  timeout 1800 "$B/llama-perplexity" -m "$mp" -f "$WIKI" -c 512 --chunks "$CH" \
      -fa off -kvu -ctk kpc4_1 -ctv f16 -t "$T" > "${base}_faoff.log" 2>&1
  faoff=$(ppl_of "${base}_faoff.log"); : "${faoff:=ERR}"

  # decode coherence. llama-cli streams the tokens then can hang on exit, so bound it tightly:
  # the 64 tokens are already in the log well before the timeout fires.
  timeout 90 "$B/llama-cli" -m "$mp" -p "Once upon a time" -n 64 \
      -fa on -ctk kpc4_1 -ctv q4_1 -t "$T" -no-cnv </dev/null > "${base}_dec.log" 2>&1
  if grep -qiE 'once upon a time' "${base}_dec.log" \
     && ! grep -qiE '\b(nan|inf)\b|GGML_ASSERT|terminate|CUDA error' "${base}_dec.log"; then
    dec=OK; else dec=CHECK; fi

  # SWA cell-recycling: opt-in (SWA_OVERFLOW=1), generates past the window so the ring pool
  # recycles non-contiguous cells. Slow on big windows/models, hence off by default.
  if [ "${SWA_OVERFLOW:-0}" = "1" ] && [ "$swa" = "1" ] && [ "$win" -gt 0 ]; then
    timeout 1800 "$B/llama-cli" -m "$mp" -p "Once upon a time" -n $(( win + 256 )) -c $(( win + 512 )) \
        -fa on -ctk kpc4_1 -ctv q4_1 -t "$T" -no-cnv </dev/null > "${base}_swa.log" 2>&1
    if grep -qiE '\b(nan|inf)\b|GGML_ASSERT|terminate' "${base}_swa.log"; then dec="${dec}/SWA!"; else dec="${dec}/SWA-ok"; fi
  fi

  tg=$(timeout 600 "$B/llama-bench" -m "$mp" -p 512 -n 128 -fa 1 \
        -ctk kpc4_1 -ctv q4_1 -t "$T" -r 2 2>/dev/null \
        | grep -E 'tg128' | grep -oE '[0-9]+\.[0-9]+' | head -1)
  : "${tg:=?}"

  printf '%-12s | %-16s | %-9s | %-9s | %-8s | %-9s | %-10s | %-8s | %-6s\n' \
    "$name" "$feat" "$kpc" "$f16" "$d" "$faoff" "$kv" "$tg" "$dec"
done

cat <<'EOF'

PASS criteria:
  dPPL    : small and positive (KPC slightly above f16; a blow-up or NaN/ERR is a failure)
  faoff   : finite and close to the fused KPC PPL (the -fa off dequant path)
  decode  : OK  (coherent text, no nan/inf/assert)
  KV-cache: KPC should be well below the f16 row's KV size (the whole point)

SWA ring-pool recycling (the path nothing has tested on a real model) is opt-in:
  SWA_OVERFLOW=1 bash scripts/kpc-arch-test.sh   # adds /SWA-ok or /SWA! to the decode column
  (generates past each SWA model's window so cells recycle non-contiguously; slow)

Tunables (env): LLAMA_BIN, WIKI, T (threads), CHUNKS (PPL chunks, default 24).
Logs: /tmp/kpcarch_<name>_{kpc,f16,faoff,dec,swa}.log
EOF
