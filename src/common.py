"""Общие константы, пути и разбор URL для всех шагов пайплайна.

Раньше разбор URL (host / split / host_of) и загрузка данных были продублированы
в каждом скрипте с мелкими расхождениями. Теперь всё в одном месте, а фолды
гарантированно одинаковые во всех шагах.

Пути можно переопределить переменными окружения:
  PHISH_DATA  - папка с train.csv / test.csv (по умолчанию data/)
  PHISH_ART   - папка артефактов (по умолчанию artifacts/)
  PHISH_SUBS  - папка файлов ответов (по умолчанию submissions/)
  PHISH_JOBS  - число потоков для моделей (по умолчанию все ядра)
"""
import os, re
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np, pandas as pd
import tldextract

SEED = 42
N_FOLDS = 5
ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get('PHISH_DATA', ROOT / 'data'))
ART = Path(os.environ.get('PHISH_ART', ROOT / 'artifacts'))
FEATS = ART / 'features'          # признаки сторонних источников, DNS, RDAP, kNN
PREDS = ART / 'preds'             # oof_*.npy / test_*.npy
CACHE = ART / 'cache'             # кэш лексических признаков (не коммитится)
SUBS = Path(os.environ.get('PHISH_SUBS', ROOT / 'submissions'))
EXT = ROOT / 'ext'                # скачанные сторонние источники
N_JOBS = int(os.environ.get('PHISH_JOBS', os.cpu_count() or 4))

for _p in (FEATS, PREDS, CACHE, SUBS):
    _p.mkdir(parents=True, exist_ok=True)

# список публичных суффиксов из поставки tldextract, без обращения в сеть
EXTRACT = tldextract.TLDExtract(suffix_list_urls=())
_SCHEME = re.compile(r'^[a-zA-Z][a-zA-Z0-9+.-]*://')
_IP = re.compile(r'[\d.]+')


def with_scheme(u):
    s = str(u).strip()
    return s if _SCHEME.match(s) else 'http://' + s


def host(u):
    """Хост в нижнем регистре; пустая строка, если URL не разбирается."""
    try: return (urlsplit(with_scheme(u)).hostname or '').lower()
    except ValueError: return ''


def host_path(u):
    """(netloc, путь + ?запрос) - вход для текстовых моделей по частям URL."""
    u = str(u).strip()
    try:
        p = urlsplit(with_scheme(u)); return (p.netloc or '').lower(), p.path + ('?' + p.query if p.query else '')
    except ValueError:
        return u.lower(), ''


@lru_cache(maxsize=None)
def registered(h):
    """Зарегистрированный домен (eTLD+1); для IP и мусора - сам хост."""
    return EXTRACT(h).top_domain_under_public_suffix or h


def is_ip(h):
    return bool(_IP.fullmatch(h)) or h.startswith('[')


@lru_cache(maxsize=1)
def load():
    """train, test, y. Кэшируется в пределах процесса."""
    tr = pd.read_csv(DATA / 'train.csv'); te = pd.read_csv(DATA / 'test.csv')
    return tr, te, tr.result.values


def folds(y):
    """Одно и то же разбиение во всех скриптах - иначе out-of-fold признаки протекают."""
    from sklearn.model_selection import StratifiedKFold
    return list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))


def save_preds(name, oof, test):
    np.save(PREDS / f'oof_{name}.npy', oof); np.save(PREDS / f'test_{name}.npy', test)


def load_preds(name):
    return np.load(PREDS / f'oof_{name}.npy'), np.load(PREDS / f'test_{name}.npy')


def write_sub(name, pred):
    pd.DataFrame({'Id': np.arange(len(pred)), 'Predicted': pred.astype(int)}).to_csv(SUBS / f'sub_{name}.csv', index=False)
