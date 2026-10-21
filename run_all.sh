#!/usr/bin/env bash
# Полный прогон решения. Нужны data/train.csv и data/test.csv с платформы.
#
# Шаги 3 и 6 (сторонние данные, DNS, RDAP) по умолчанию пропускаются: их результаты
# лежат в artifacts/features/ и меняются со временем. Пересчитать заново: REBUILD_EXT=1 ./run_all.sh
# Переменные: PY - интерпретатор, PHISH_JOBS - потоки (для бит-в-бит повторения исходного прогона: 8).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
S=src
step() { echo; echo "=== $*"; }

step "1-2: текстовые модели первого уровня"
$PY $S/text_models.py
if [ "${REBUILD_EXT:-0}" = "1" ]; then
  step "3: признаки сторонних источников (нужна папка ext/)"; $PY $S/ext_feats.py
fi
step "4: промежуточный стек для псевдоразметки"
$PY $S/stack_lgb.py stack4 --ext
step "5: текстовые модели с псевдоразметкой"
$PY $S/text_models.py --pl stack4 --thr 0.99 --suffix _pl
if [ "${REBUILD_EXT:-0}" = "1" ]; then
  step "6: DNS и RDAP (живые проверки)"; $PY $S/dns_feats.py; $PY $S/rdap_feats.py
fi
step "6: ближайшие соседи"
$PY $S/knn_feats.py

step "7: стеки второго уровня"
B="tfidf_pl host_char_pl path_char_pl full_sgd_pl"
COMMON="--ext --dns --rdap ${STACK_FLAGS:-}"
$PY $S/stack_lgb.py stack9s   $COMMON --seeds 5 --base $B
$PY $S/stack_lgb.py stack9cat $COMMON --cat     --base $B
$PY $S/stack_lgb.py stack10   $COMMON --knn     --base $B
$PY $S/text_models.py --pl stack9s --thr 0.98 --suffix _pl2
$PY $S/stack_lgb.py stack11   $COMMON --base tfidf_pl2 host_char_pl2 path_char_pl2 full_sgd_pl2

step "8: смесь стеков"
$PY $S/blend.py blend3 stack9s stack9cat stack10
$PY $S/blend.py blend4 stack9s stack9cat stack10 stack11
echo "готово: sub_blend4.csv"
