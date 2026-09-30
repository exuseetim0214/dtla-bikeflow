#!/usr/bin/env bash
set -euo pipefail
streamlit run src/bikeflow/dashboard.py "$@"

