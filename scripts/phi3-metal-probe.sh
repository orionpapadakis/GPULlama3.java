#!/usr/bin/env bash
# Probe the Phi-3 FP16 Metal hang on the macOS CI runner.
#
# Runs the CI row's own command, samples the JVM's stacks and the machine's memory
# pressure while it runs, and stops after a bounded wall clock. Every artifact lands
# in ./phi3-probe-<label>/ so a run can be handed back whole.
#
# Usage: scripts/phi3-metal-probe.sh <label> [extra llama-tornado flags...]
#   e.g. scripts/phi3-metal-probe.sh baseline
#        scripts/phi3-metal-probe.sh mem11 --gpu-memory 11GB
set -uo pipefail

LABEL="${1:?usage: $0 <label> [extra flags...]}"; shift
MODEL="${MODEL:-/opt/models/Phi-3-mini-4k-instruct-fp16.gguf}"
TOKENS="${TOKENS:-32}"
# A prompt long enough to fill every position leaves no position to sample at, which
# is how a position-dependent stall is told apart from a sampling-dependent one.
PROMPT="${PROMPT:-Tell me a joke}"
BUDGET_SECONDS="${BUDGET_SECONDS:-420}"
OUT="phi3-probe-${LABEL}"

mkdir -p "$OUT"
: > "$OUT/run.log"

echo "model=$MODEL tokens=$TOKENS budget=${BUDGET_SECONDS}s prompt_words=$(echo "$PROMPT" | wc -w) extra=$*" | tee "$OUT/params.txt"

# Line-buffer the launcher's stdout so the log shows when a token actually appeared,
# not when the pipe was flushed at kill time.
( PYTHONUNBUFFERED=1 ./llama-tornado \
    --gpu \
    --metal \
    --model "$MODEL" \
    --prompt "$PROMPT" \
    -n "$TOKENS" \
    --verbose-init \
    "$@" 2>&1 \
  | while IFS= read -r line; do printf '%s %s\n' "$(date -u +%H:%M:%S)" "$line"; done \
  >> "$OUT/run.log" ) &
WRAPPER=$!

# Sample until the wrapper exits or the budget runs out.
for ((t = 0; t < BUDGET_SECONDS; t += 20)); do
    kill -0 "$WRAPPER" 2>/dev/null || break
    JAVA_PID=$(pgrep -f 'java.*(gpullama3|LlamaApp)' | head -1)
    {
        printf '=== t=%ss %s ===\n' "$t" "$(date -u +%H:%M:%S)"
        printf 'java_pid=%s\n' "${JAVA_PID:-none}"
        if [[ -n "${JAVA_PID:-}" ]]; then
            ps -o rss=,vsz=,%cpu=,state= -p "$JAVA_PID"
        fi
        vm_stat | grep -E 'Pages (free|swapped|active)|Swapins|Swapouts'
        sysctl -n vm.swapusage
        # Where the anonymous gigabytes are. ps RSS undercounts a Metal buffer on
        # unified memory, so ask for the process footprint and the region summary too.
        if [[ -n "${JAVA_PID:-}" ]]; then
            footprint -p "$JAVA_PID" 2>&1 | head -25
            vmmap --summary "$JAVA_PID" 2>&1 | head -30
        fi
    } >> "$OUT/memory.txt" 2>&1
    if [[ -n "${JAVA_PID:-}" ]]; then
        {
            printf '\n\n=== stacks t=%ss ===\n' "$t"
            jcmd "$JAVA_PID" Thread.print 2>&1
        } >> "$OUT/stacks.txt"
    fi
    sleep 20
done

if kill -0 "$WRAPPER" 2>/dev/null; then
    echo "HUNG: still running after ${BUDGET_SECONDS}s" | tee -a "$OUT/params.txt"
    # Kill the process group so the java grandchild does not survive the wrapper,
    # which is the leak .github/actions/run-inference had to fix.
    JAVA_PID=$(pgrep -f 'java.*(gpullama3|LlamaApp)' | head -1)
    [[ -n "${JAVA_PID:-}" ]] && kill -QUIT "$JAVA_PID" 2>/dev/null && sleep 5
    pkill -f 'java.*(gpullama3|LlamaApp)' 2>/dev/null
    kill -TERM "$WRAPPER" 2>/dev/null
    sleep 3
    pkill -9 -f 'java.*(gpullama3|LlamaApp)' 2>/dev/null
else
    wait "$WRAPPER"
    echo "exit=$? " | tee -a "$OUT/params.txt"
fi

echo "artifacts in $OUT/: run.log stacks.txt memory.txt params.txt"
