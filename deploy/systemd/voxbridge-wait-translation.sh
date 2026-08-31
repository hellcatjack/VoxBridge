#!/usr/bin/env bash
set -Eeuo pipefail

models_url="${VOXBRIDGE_TRANSLATION_MODELS_URL:-http://127.0.0.1:8001/v1/models}"
expected_model="${VOXBRIDGE_TRANSLATION_MODEL:-tencent/HY-MT1.5-1.8B-GGUF:Q8_0}"
timeout_sec="${VOXBRIDGE_TRANSLATION_READY_TIMEOUT_SEC:-120}"

if ! [[ "${timeout_sec}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid translation readiness timeout: ${timeout_sec}" >&2
  exit 2
fi
if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required for the translation readiness check" >&2
  exit 2
fi

deadline=$((SECONDS + timeout_sec))
while ((SECONDS < deadline)); do
  response="$(curl --fail --silent --show-error --max-time 2 "${models_url}" 2>/dev/null || true)"
  if [[ -n "${response}" && "${response}" == *"${expected_model}"* ]]; then
    echo "Translation API ready: ${expected_model}"
    exit 0
  fi
  sleep 1
done

echo "Translation API did not expose ${expected_model} within ${timeout_sec}s" >&2
exit 1
