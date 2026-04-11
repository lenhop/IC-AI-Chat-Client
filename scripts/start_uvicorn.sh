#!/usr/bin/env bash
#
# Start FastAPI with uvicorn after safely releasing target LISTEN port.
# LLM service unified API: POST /v1/chat/stream.
#
# Dependencies:
# - bash (macOS / common Linux distributions)
# - lsof
#
# Exit codes:
#   1  : dependency missing or invalid argument
#   2  : timed out while releasing occupied port
#   3  : timed out while waiting LLM service startup
#  10  : failed to enter repository root
#  11+ : uvicorn process exits with its own status code (exec replaces shell)

set -u

log_info() {
  echo "[start_uvicorn][INFO] $*"
}

log_warn() {
  echo "[start_uvicorn][WARN] $*" >&2
}

log_error() {
  echo "[start_uvicorn][ERROR] $*" >&2
}

require_dependency() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    log_error "Missing dependency: ${cmd}. Please install it and retry."
    exit 1
  fi
}

validate_port() {
  local port="$1"
  if [[ ! "${port}" =~ ^[0-9]+$ ]]; then
    log_error "UVICORN_PORT must be an integer, got: ${port}"
    exit 1
  fi
  if (( port < 1 || port > 65535 )); then
    log_error "UVICORN_PORT out of range [1, 65535], got: ${port}"
    exit 1
  fi
}

get_listen_pids() {
  local port="$1"
  # LISTEN-only query avoids killing non-listening connections on same port.
  lsof -nP -iTCP:"${port}" -sTCP:LISTEN -t 2>/dev/null | awk '!seen[$0]++'
}

kill_pid_list() {
  local signal="$1"
  local pids="$2"
  local pid=""
  while IFS= read -r pid; do
    [[ -z "${pid}" ]] && continue
    kill "-${signal}" "${pid}" >/dev/null 2>&1 || true
  done <<< "${pids}"
}

release_port_or_timeout() {
  local port="$1"
  local timeout_seconds="$2"
  local term_wait_seconds="$3"
  local poll_seconds="$4"
  local deadline=$((SECONDS + timeout_seconds))

  while true; do
    local pids
    pids="$(get_listen_pids "${port}")"
    if [[ -z "${pids}" ]]; then
      log_info "Port ${port} is available."
      return 0
    fi

    log_warn "Port ${port} in use by PID(s): ${pids//$'\n'/,}. Sending SIGTERM."
    kill_pid_list "TERM" "${pids}"
    sleep "${term_wait_seconds}"

    pids="$(get_listen_pids "${port}")"
    if [[ -z "${pids}" ]]; then
      log_info "Port ${port} released after SIGTERM."
      return 0
    fi

    log_warn "PID(s) still listening: ${pids//$'\n'/,}. Sending SIGKILL."
    kill_pid_list "KILL" "${pids}"
    sleep "${poll_seconds}"

    if (( SECONDS >= deadline )); then
      log_error "Timeout: unable to release port ${port} within ${timeout_seconds}s."
      return 2
    fi
  done
}

wait_for_port_listen_or_timeout() {
  local port="$1"
  local timeout_seconds="$2"
  local poll_seconds="$3"
  local deadline=$((SECONDS + timeout_seconds))

  while true; do
    local pids
    pids="$(get_listen_pids "${port}")"
    if [[ -n "${pids}" ]]; then
      log_info "Port ${port} is now listening (PID(s): ${pids//$'\n'/,})."
      return 0
    fi
    if (( SECONDS >= deadline )); then
      log_error "Timeout: port ${port} did not start listening within ${timeout_seconds}s."
      return 3
    fi
    sleep "${poll_seconds}"
  done
}

main() {
  require_dependency "lsof"
  require_dependency "python"

  local host="${UVICORN_HOST:-0.0.0.0}"
  local port="${UVICORN_PORT:-8000}"
  local start_llm_service="${START_LLM_SERVICE:-1}"
  local llm_host="${LLM_SERVICE_HOST:-0.0.0.0}"
  local llm_port="${LLM_SERVICE_PORT:-8001}"
  local llm_startup_wait_seconds="${LLM_SERVICE_STARTUP_WAIT_SECONDS:-20}"
  local llm_log_file="${LLM_SERVICE_LOG_FILE:-/tmp/icai-llm-service.log}"
  local release_timeout_seconds="${START_UVICORN_RELEASE_TIMEOUT_SECONDS:-20}"
  local term_wait_seconds="${START_UVICORN_TERM_WAIT_SECONDS:-1}"
  local poll_seconds="${START_UVICORN_POLL_SECONDS:-1}"

  validate_port "${port}"
  validate_port "${llm_port}"
  if [[ "${port}" == "${llm_port}" ]]; then
    log_error "UVICORN_PORT and LLM_SERVICE_PORT must differ (both are ${port})."
    exit 1
  fi

  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  local repo_root
  repo_root="$(cd "${script_dir}/.." && pwd)"
  cd "${repo_root}" || {
    log_error "Cannot enter repository root: ${repo_root}"
    exit 10
  }

  if [[ "${start_llm_service}" == "1" ]]; then
    release_port_or_timeout "${llm_port}" "${release_timeout_seconds}" "${term_wait_seconds}" "${poll_seconds}" || exit $?
    log_info "Starting LLM service: host=${llm_host}, port=${llm_port}, log=${llm_log_file}"
    nohup python -m uvicorn app.llm_service.main:app --host "${llm_host}" --port "${llm_port}" > "${llm_log_file}" 2>&1 &
    wait_for_port_listen_or_timeout "${llm_port}" "${llm_startup_wait_seconds}" "${poll_seconds}" || exit $?
  else
    log_info "Skipping LLM service startup (START_LLM_SERVICE=${start_llm_service})."
  fi

  release_port_or_timeout "${port}" "${release_timeout_seconds}" "${term_wait_seconds}" "${poll_seconds}" || exit $?

  log_info "Starting uvicorn: host=${host}, port=${port}, reload=true"
  exec python -m uvicorn app.main:app --host "${host}" --port "${port}" --reload
}

main "$@"
