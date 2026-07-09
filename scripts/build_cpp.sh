#!/usr/bin/env bash
# build_cpp.sh — configure + build LOB-Core Component C (lob_bridge) with cmake.
#
# Project AURA / LOB-Core. Builds into build/ using the project's .venv so the
# pybind11 module binds against the correct Python ABI, then copies the resulting
# lob_bridge*.so next to the build dir and onto PYTHONPATH (build/) for import.
#
# Usage:  scripts/build_cpp.sh [--clean]
set -euo pipefail

# Resolve repo root (this script lives in scripts/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

VENV_PY="${ROOT}/.venv/bin/python"
BUILD_DIR="${ROOT}/build"

if [[ ! -x "${VENV_PY}" ]]; then
  echo "ERROR: venv python not found at ${VENV_PY}" >&2
  exit 1
fi

if [[ "${1:-}" == "--clean" ]]; then
  echo "[build] cleaning ${BUILD_DIR}"
  rm -rf "${BUILD_DIR}"
fi

# Locate pybind11's cmake package dir from the venv.
PYBIND11_DIR="$("${VENV_PY}" -m pybind11 --cmakedir)"
echo "[build] pybind11_DIR = ${PYBIND11_DIR}"
echo "[build] python       = ${VENV_PY}"

# Configure.
cmake -S "${ROOT}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR="${PYBIND11_DIR}" \
  -DPython3_EXECUTABLE="${VENV_PY}"

# Build all targets.
cmake --build "${BUILD_DIR}" --config Release -j"$(sysctl -n hw.ncpu)"

# Surface the built artifacts. pybind11 emits lob_bridge.<abi>.so in build/.
SO="$(find "${BUILD_DIR}" -name 'lob_bridge*.so' -maxdepth 3 | head -n1 || true)"
if [[ -z "${SO}" ]]; then
  echo "ERROR: lob_bridge*.so not found under ${BUILD_DIR}" >&2
  exit 1
fi
if [[ "$(dirname "${SO}")" != "${BUILD_DIR}" ]]; then
  cp -f "${SO}" "${BUILD_DIR}/"
fi
echo "[build] module: ${SO}"
echo "[build] producer_demo: ${BUILD_DIR}/producer_demo"
echo "[build] OK — add ${BUILD_DIR} to PYTHONPATH to 'import lob_bridge'"
