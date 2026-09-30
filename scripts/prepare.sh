#!/usr/bin/env bash
set -euo pipefail
python -m bikeflow audit
python -m bikeflow prepare "$@"

