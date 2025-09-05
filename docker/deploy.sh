set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

docker build --no-cache \
  -f "${SCRIPT_DIR}/Dockerfile.deploy" \
  -t supervisely/deim:dev-deploy \
  "$PROJECT_ROOT"

docker push supervisely/deim:dev-deploy