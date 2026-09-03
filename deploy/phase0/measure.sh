#!/bin/bash
# ============================================================
# Phase 0 — measure where a hanging request gets cut
# ============================================================
# Answers the one number no Google document states: how long may a single
# Agent Runtime query hang before something severs it? The audit pipeline runs
# for minutes to hours, so this decides whether the design is synchronous,
# polling, or fire-and-notify.
#
#   ./measure.sh <base-url>            60 300 900 1800, plus a matching stream
#   ./measure.sh <base-url> 60 120     only those durations
#
# <base-url> is whatever fronts the probe container:
#   http://127.0.0.1:8099              local sanity run
#   https://<region>-aiplatform...     the deployed reasoningEngine endpoint
#
# The important trick is the second half of each round. When the client stops
# getting an answer it has learned exactly one thing -- that *it* stopped
# waiting. So after every hang the script calls `probe_log`, which reads the
# server's own record back:
#
#   hang_completed + same instance -> only the connection died; the work
#                                     survived, so fire-and-notify is viable
#   hang_cancelled                 -> the platform propagated a cancel
#   no entry, or a new instance    -> the container itself was replaced
#
# Read-only against the service. Prints a summary table at the end.

set -uo pipefail

BASE="${1:-}"
if [ -z "$BASE" ]; then
    echo "usage: ./measure.sh <base-url> [seconds ...]" >&2
    exit 2
fi
shift

DURATIONS=("$@")
[ ${#DURATIONS[@]} -eq 0 ] && DURATIONS=(60 300 900 1800)

# Two very different targets, and the difference is not cosmetic.
#
#   local   http://127.0.0.1:8099
#           talks to the container directly: contract routes, body field
#           `class_method`, no auth. Measures the container.
#
#   deployed  https://<region>-aiplatform.googleapis.com/v1/projects/.../
#             reasoningEngines/<id>
#           goes through the platform: `:query` / `:streamQuery`, body field
#           `classMethod`, bearer token. Measures the platform *and* the
#           container, which is the number Phase 0 actually needs -- a cut
#           imposed by the proxy never reaches the container at all.
#
# Both are worth running. If the local one survives 30 minutes and the
# deployed one dies at five, the ceiling belongs to the platform.
case "$BASE" in
    *reasoningEngines/*)
        MODE=deployed
        UNARY="$BASE:query"
        STREAM="$BASE:streamQuery"
        METHOD_KEY=classMethod
        ;;
    *)
        MODE=local
        UNARY="$BASE/api/reasoning_engine"
        STREAM="$BASE/api/stream_reasoning_engine"
        METHOD_KEY=class_method
        ;;
esac

# Fetched once. Access tokens last an hour, and a 1800s hang plus its ceiling
# can outlive that -- refresh per call so a long round does not die of a stale
# token and get misread as a platform timeout.
auth_header() {
    [ "$MODE" = deployed ] || return 0
    printf 'Authorization: Bearer %s' "$(gcloud auth print-access-token 2>/dev/null)"
}

# Never let curl be the thing that gives up first -- that would measure curl,
# not the platform. Each call gets a ceiling well past its own duration.
call() {
    local endpoint="$1" method="$2" input="$3" max="$4"
    local body="{\"$METHOD_KEY\":\"$method\",\"input\":$input}"
    if [ "$MODE" = deployed ]; then
        curl -sS --no-buffer --max-time "$max" \
            -X POST "$endpoint" \
            -H "$(auth_header)" \
            -H 'Content-Type: application/json' \
            -d "$body" 2>&1
    else
        curl -sS --no-buffer --max-time "$max" \
            -X POST "$endpoint" \
            -H 'Content-Type: application/json' \
            -d "$body" 2>&1
    fi
}

# The container answers compact JSON; the platform re-serialises it pretty-
# printed, so `"instance":"x"` only matches locally. Tolerate the space, or
# every deployed round reads as "container replaced" and the whole table lies.
instance_of() {
    grep -oE '"instance":[[:space:]]*"[^"]+"' | head -1 | cut -d'"' -f4
}

RESULTS=()

echo "============================================================"
echo " Phase 0 timeout measurement"
echo " target: $BASE"
echo " mode:   $MODE"
echo "============================================================"
echo

printf 'Reachability ... '
HELLO="$(call "$UNARY" hello '{"message":"measure.sh"}' 30)"
if echo "$HELLO" | grep -q '"ok":[[:space:]]*true'; then
    START_INSTANCE="$(echo "$HELLO" | instance_of)"
    echo "✅ instance $START_INSTANCE"
else
    echo "❌ could not reach the probe"
    echo "$HELLO" | head -5 | sed 's/^/        /'
    exit 1
fi
echo

for SECONDS_WANTED in "${DURATIONS[@]}"; do
    # A generous margin so a clean finish is never mistaken for a cut.
    CEILING=$((SECONDS_WANTED + 120))

    echo "------------------------------------------------------------"
    echo " hang ${SECONDS_WANTED}s  (client will wait up to ${CEILING}s)"
    echo "------------------------------------------------------------"

    T0=$(date +%s)
    OUT="$(call "$UNARY" hang "{\"seconds\":$SECONDS_WANTED}" "$CEILING")"
    T1=$(date +%s)
    CLIENT_ELAPSED=$((T1 - T0))

    if echo "$OUT" | grep -q '"ok":[[:space:]]*true'; then
        CLIENT_VERDICT="answered after ${CLIENT_ELAPSED}s"
        echo "  client: ✅ $CLIENT_VERDICT"
    else
        CLIENT_VERDICT="cut at ${CLIENT_ELAPSED}s"
        echo "  client: ❌ $CLIENT_VERDICT"
        echo "$OUT" | head -3 | sed 's/^/          /'
    fi

    # The half that actually matters. Give the server a moment to finish
    # writing its own record before asking for it.
    sleep 3
    LOG="$(call "$UNARY" probe_log '{"limit":8}' 60)"
    NOW_INSTANCE="$(echo "$LOG" | instance_of)"

    if [ -z "$NOW_INSTANCE" ]; then
        # probe_log itself failed. Say so rather than reporting a phantom
        # container swap -- an empty string is not a new instance id.
        SERVER_VERDICT="probe_log unreachable: $(echo "$LOG" | tr -d '\n' | cut -c1-120)"
    elif [ "$NOW_INSTANCE" != "$START_INSTANCE" ]; then
        SERVER_VERDICT="container replaced ($START_INSTANCE -> $NOW_INSTANCE)"
        START_INSTANCE="$NOW_INSTANCE"
    elif echo "$LOG" | grep -q 'hang_cancelled'; then
        SERVER_VERDICT="platform propagated a cancel"
    elif echo "$LOG" | grep -q 'hang_completed'; then
        SERVER_VERDICT="work completed on the server"
    else
        SERVER_VERDICT="still running or no record"
    fi
    echo "  server: $SERVER_VERDICT"

    RESULTS+=("hang ${SECONDS_WANTED}s|$CLIENT_VERDICT|$SERVER_VERDICT")
    echo
done

# A stream that keeps producing bytes is a different case from a unary request
# that goes quiet: idle timeouts usually only punish silence. If the stream
# outlives the hang, progress streaming becomes the design and the polling
# fallback is unnecessary.
LONGEST="${DURATIONS[${#DURATIONS[@]}-1]}"
echo "------------------------------------------------------------"
echo " stream ${LONGEST}s, heartbeat every 10s"
echo "------------------------------------------------------------"
T0=$(date +%s)
TICKS="$(call "$STREAM" stream_query "{\"seconds\":$LONGEST,\"interval_seconds\":10}" $((LONGEST + 120)) | grep -c '"tick"')"
T1=$(date +%s)
STREAM_ELAPSED=$((T1 - T0))
EXPECTED=$((LONGEST / 10))
echo "  received $TICKS heartbeats in ${STREAM_ELAPSED}s (expected ~$EXPECTED)"
RESULTS+=("stream ${LONGEST}s|$TICKS/$EXPECTED heartbeats in ${STREAM_ELAPSED}s|-")
echo

echo "------------------------------------------------------------"
echo " Egress"
echo "------------------------------------------------------------"
call "$UNARY" egress '{}' 90 | sed 's/^/  /'
echo
echo

echo "============================================================"
echo " Summary"
echo "============================================================"
printf '%-16s %-28s %s\n' "PROBE" "CLIENT SAW" "SERVER SAYS"
for row in "${RESULTS[@]}"; do
    IFS='|' read -r a b c <<< "$row"
    printf '%-16s %-28s %s\n' "$a" "$b" "$c"
done
echo
echo "Record these in deploy/phase0/README.md before moving to Phase 1."
