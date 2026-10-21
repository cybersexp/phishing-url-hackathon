#!/usr/bin/env bash
# Смоук-тест всего пайплайна на синтетических данных (~2-3 минуты).
# Проверяет, что все шаги run_all.sh запускаются и связаны друг с другом; качество модели не проверяет.
# Внешние источники, DNS и RDAP заменены заглушками, сеть не нужна.
set -euo pipefail
cd "$(dirname "$0")/.."
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/data"
PY=${PY:-python3}
$PY tests/make_synth.py "$TMP/data" ${N_TRAIN:-3000} ${N_TEST:-800}
PHISH_DATA="$TMP/data" PHISH_ART="$TMP/artifacts" PHISH_SUBS="$TMP/submissions" PY=$PY ./run_all.sh
test -s "$TMP/submissions/sub_blend4.csv"
echo "smoke OK"
