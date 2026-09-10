#!/bin/bash
# ============================================================
# CCTV Audit Agent — pre-push self-check
# ============================================================
# Answers one question before you stop for the day: is the work that exists on
# this machine also safely on GitHub, and does it still run?
#
# It exists because a push once "succeeded" while carrying nothing but
# deletions -- `git commit` had run before `git add`, and every local signal
# (a commit in the log, a clean push) looked correct. Only a file-by-file
# comparison against the remote caught it.
#
#   ./check.sh          six fast checks, ~10s, no network beyond one git fetch
#   ./check.sh --deep   also re-clones the branch from GitHub, diffs it against
#                       this directory and runs the tests inside the clone
#
# Read-only by design. It never commits, pushes or edits anything: a checking
# tool that also mutates is just a new way to lose work. Exit code is 0 when
# everything passed, 1 otherwise, so it can be wired into CI or a hook later.

# Deliberately no `-e`. The whole job is to collect failures; aborting on the
# first one would hide the other five.
set -uo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

DEEP=false
for arg in "$@"; do
    case "$arg" in
        --deep) DEEP=true ;;
        -h|--help)
            sed -n '3,19p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Unknown option: $arg (try --deep or --help)"
            exit 2
            ;;
    esac
done

TOTAL=6
STEP=0
FAILED=0
FIXES=()

# ------------------------------------------------------------
# Output helpers
# ------------------------------------------------------------
step() {
    STEP=$((STEP + 1))
    local label="$1"
    local width=$((34 - ${#label}))
    [ "$width" -lt 1 ] && width=1
    printf '[%d/%d] %s %s ' "$STEP" "$TOTAL" "$label" \
        "$(printf '%*s' "$width" '' | tr ' ' '.')"
}

pass() { echo "✅ ${1:-ok}"; }
warn() { echo "⚠️  $1"; }

fail() {
    echo "❌ $1"
    FAILED=$((FAILED + 1))
}

# Indents a block of detail lines under the check that produced them.
detail() { sed 's/^/        /'; }

# Remembers the command that fixes a failed check, printed once at the end.
suggest() { FIXES+=("$1"); }

echo "============================================================"
echo " Pre-push self-check"
echo "============================================================"
echo

# ------------------------------------------------------------
# Python environment
# ------------------------------------------------------------
# Creating it is start_web.sh's job, not this script's -- two places building
# the same venv is two places to drift.
if [ ! -d ".venv" ]; then
    echo "❌ No .venv here. Run ./start_web.sh once to build it, then re-run this."
    exit 2
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# ------------------------------------------------------------
# 1. Unit tests
# ------------------------------------------------------------
step "Unit tests"
PYTEST_OUT="$(python -m pytest -q 2>&1)"
if [ $? -eq 0 ]; then
    pass "$(echo "$PYTEST_OUT" | tail -1)"
else
    fail "$(echo "$PYTEST_OUT" | tail -1)"
    echo "$PYTEST_OUT" | grep -E '^(FAILED|ERROR)' | head -10 | detail
    suggest "python -m pytest -q            # see the failures in full"
fi

# ------------------------------------------------------------
# 2. No secrets committed
# ------------------------------------------------------------
# Two separate mistakes to catch: a secret-bearing path that got tracked
# despite .gitignore (`git add -f` does that silently), and a credential pasted
# into a file that is legitimately tracked.
step "No secrets committed"
SECRET_PATHS="$(git ls-files | grep -E '(^|/)\.env$|^auth/|(^|/)checkpoints/|\.pem$|(^|/)id_(rsa|ed25519)$' || true)"

# These patterns do not match their own source text here (each is broken by a
# regex metacharacter where the pattern expects a literal), so this file is
# scanned like any other rather than excluded from its own check.
SECRET_CONTENT="$(git grep -I -l -E \
    'AIza[0-9A-Za-z_-]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY|"private_key"[[:space:]]*:' \
    -- . 2>/dev/null || true)"

if [ -z "$SECRET_PATHS" ] && [ -z "$SECRET_CONTENT" ]; then
    pass
else
    fail "credentials are tracked by git"
    [ -n "$SECRET_PATHS" ] && printf '%s\n' "$SECRET_PATHS" | detail
    [ -n "$SECRET_CONTENT" ] && printf '%s  (credential-shaped content)\n' "$SECRET_CONTENT" | detail
    suggest "git rm --cached <file>         # untrack it; note it STAYS in history"
fi

# ------------------------------------------------------------
# 3. Working tree clean
# ------------------------------------------------------------
step "Working tree clean"
DIRTY="$(git status --porcelain --untracked-files=no)"
if [ -z "$DIRTY" ]; then
    pass
else
    fail "$(printf '%s\n' "$DIRTY" | wc -l) uncommitted file(s)"
    printf '%s\n' "$DIRTY" | head -15 | detail
    suggest "git add -A && git commit -m \"...\""
fi

# ------------------------------------------------------------
# 4. No stray untracked files
# ------------------------------------------------------------
# A new module that was never `git add`ed imports fine locally and simply does
# not exist for anyone else.
step "No stray untracked files"
STRAY="$(git ls-files --others --exclude-standard)"
if [ -z "$STRAY" ]; then
    pass
else
    fail "$(printf '%s\n' "$STRAY" | wc -l) file(s) git has never seen"
    printf '%s\n' "$STRAY" | head -15 | detail
    suggest "git add <file>                 # or add it to .gitignore on purpose"
fi

# ------------------------------------------------------------
# 5. No source hidden by .gitignore
# ------------------------------------------------------------
# The nastiest failure mode: a too-broad ignore rule quietly drops source from
# every clone while everything keeps working here. `--directory` collapses
# wholly-ignored trees so we do not walk all of .venv.
step "No source hidden by .gitignore"
HIDDEN=""
while IFS= read -r entry; do
    [ -z "$entry" ] && continue
    case "$entry" in
        # The ignores we mean: environments, caches, scratch, secrets, output.
        # Drifting from .gitignore only costs a false alarm here, never a miss.
        .venv/|.git/|__pycache__/|*/__pycache__/|.pytest_cache/|.work/|auth/|node_modules/) continue ;;
        cctv_audit/.adk|cctv_audit/.adk/|cctv_audit/checkpoints/) continue ;;
        cctv_audit/.env|.DS_Store|BRD.md|SDD.md) continue ;;
    esac
    if [ -d "$entry" ]; then
        found="$(find "$entry" -type f \
            \( -name '*.py' -o -name '*.yaml' -o -name '*.yml' -o -name '*.sh' -o -name '*.md' \) \
            2>/dev/null | head -5)"
        [ -n "$found" ] && HIDDEN="${HIDDEN}${found}"$'\n'
    else
        case "$entry" in
            *.py|*.yaml|*.yml|*.sh|*.md) HIDDEN="${HIDDEN}${entry}"$'\n' ;;
        esac
    fi
done <<< "$(git ls-files --others --ignored --exclude-standard --directory)"

HIDDEN="$(printf '%s' "$HIDDEN" | sed '/^$/d')"
if [ -z "$HIDDEN" ]; then
    pass
else
    fail "source files excluded from git"
    printf '%s\n' "$HIDDEN" | detail
    suggest "git check-ignore -v <file>     # shows which .gitignore line hides it"
fi

# ------------------------------------------------------------
# 6. In sync with GitHub
# ------------------------------------------------------------
step "In sync with GitHub"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
UPSTREAM="$(git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' 2>/dev/null)"
if [ -z "$UPSTREAM" ]; then
    # Not "nothing to compare against" -- this is the state where the work only
    # exists here, which is exactly what the script is for.
    fail "branch '$BRANCH' has never been pushed"
    suggest "git push -u origin $BRANCH"
elif ! git fetch --quiet origin 2>/dev/null; then
    warn "skipped (could not reach GitHub)"
else
    COUNTS="$(git rev-list --left-right --count "$UPSTREAM"...HEAD 2>/dev/null)"
    BEHIND="$(echo "$COUNTS" | cut -f1)"
    AHEAD="$(echo "$COUNTS" | cut -f2)"
    if [ "${AHEAD:-0}" -eq 0 ] && [ "${BEHIND:-0}" -eq 0 ]; then
        pass "$BRANCH == $UPSTREAM"
    else
        MSG=""
        [ "${AHEAD:-0}" -gt 0 ] && MSG="$AHEAD commit(s) not pushed"
        [ "${BEHIND:-0}" -gt 0 ] && MSG="${MSG:+$MSG, }$BEHIND commit(s) not pulled"
        fail "$MSG"
        [ "${AHEAD:-0}" -gt 0 ] && git log --oneline "$UPSTREAM"..HEAD | head -5 | detail
        [ "${AHEAD:-0}" -gt 0 ] && suggest "git push"
        [ "${BEHIND:-0}" -gt 0 ] && suggest "git pull --rebase"
    fi
fi

# ------------------------------------------------------------
# --deep: prove the code survives leaving this machine
# ------------------------------------------------------------
# Everything above trusts this repository's own bookkeeping. This does not: it
# fetches the branch as a stranger would and checks that what comes back is
# byte-identical and still passes its tests. Worth the 20-30s before a PR or a
# handover; overkill for a routine end-of-day check.
if [ "$DEEP" = true ]; then
    echo
    echo "------------------------------------------------------------"
    echo " Deep check: re-clone from GitHub and verify"
    echo "------------------------------------------------------------"
    CLONE_DIR="$(mktemp -d)"
    trap 'rm -rf "$CLONE_DIR"' EXIT

    REMOTE_URL="$(git remote get-url origin 2>/dev/null)"
    printf '  Cloning %s ... ' "$BRANCH"
    if git clone -q --branch "$BRANCH" --single-branch "$REMOTE_URL" "$CLONE_DIR/repo" 2>/dev/null; then
        echo "ok"

        # The exclude list is read out of .gitignore, not typed out here. It
        # used to be a hand-kept copy and it fell behind exactly the way a
        # second copy does: `deploy/demovideo/assets/` was added to .gitignore
        # and not here, so every --deep run reported a difference that was not
        # one. A check that cries wolf on every run stops being read.
        #
        # `diff --exclude` matches basenames, so a pattern like
        # `cctv_audit/.env` contributes `.env` -- which is what the old
        # hand-written list said too.
        DIFF_EXCLUDES=(--exclude=.git)
        while IFS= read -r pattern; do
            case "$pattern" in ''|'#'*) continue ;; esac
            pattern="${pattern%/}"
            DIFF_EXCLUDES+=("--exclude=${pattern##*/}")
        done < "$DIR/.gitignore"

        printf '  Comparing against this directory ... '
        DIFF_OUT="$(diff -r --brief "${DIFF_EXCLUDES[@]}" \
            "$CLONE_DIR/repo" "$DIR" 2>&1)"
        if [ -z "$DIFF_OUT" ]; then
            echo "✅ identical"
        else
            fail "GitHub and this directory differ"
            printf '%s\n' "$DIFF_OUT" | head -15 | detail
            suggest "git status                     # something here is not on GitHub"
        fi

        printf '  Running the tests inside the clone ... '
        CLONE_OUT="$(cd "$CLONE_DIR/repo" && python -m pytest -q 2>&1)"
        if [ $? -eq 0 ]; then
            echo "✅ $(echo "$CLONE_OUT" | tail -1)"
        else
            fail "the code on GitHub does not pass its own tests"
            echo "$CLONE_OUT" | tail -10 | detail
        fi
    else
        fail "could not clone '$BRANCH' from GitHub"
        suggest "git push -u origin $BRANCH"
    fi
fi

# ------------------------------------------------------------
# Verdict
# ------------------------------------------------------------
echo
echo "------------------------------------------------------------"
if [ "$FAILED" -eq 0 ]; then
    echo "✅ All checks passed. Safe to stop."
else
    echo "❌ $FAILED check(s) failed. Fix the ❌ above before you stop."
    for fix in "${FIXES[@]}"; do
        echo "   $fix"
    done
fi

exit $(( FAILED > 0 ))
