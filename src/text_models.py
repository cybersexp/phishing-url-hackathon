"""Шаги 1, 2, 5 и 5а. Линейные текстовые модели первого уровня на TF-IDF по URL.

Заменяет четыре почти одинаковых скрипта исходной версии (baseline_tfidf.py,
tfidf_models.py, text_pl.py, text_pl2.py): описание моделей теперь одно,
а режим задаётся флагами.

  tfidf      - символы 1-5 + слова 1-2 по всему URL, логистическая регрессия
  host_char  - символьные n-граммы только по хосту
  path_char  - только по пути и строке запроса
  full_sgd   - широкие символьные n-граммы 1-6 по всему URL, SGD

Запуск:
  python src/text_models.py                                   # шаги 1-2, без псевдоразметки
  python src/text_models.py --pl stack4  --thr 0.99 --suffix _pl    # шаг 5
  python src/text_models.py --pl stack9s --thr 0.98 --suffix _pl2   # шаг 5а

Псевдоразметка: в обучающую часть каждого фолда добавляются строки теста с надёжной
меткой - из размеченных внешних датасетов (x_lab_url) либо там, где стек --pl уверен
не меньше чем на thr. Валидационная часть фолда остаётся чистой, оценка - по меткам train.

Модели считаются параллельно (по процессу на модель). Результат: oof/test_<имя><suffix>.npy.
"""
import argparse, numpy as np, pandas as pd, scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score

from common import SEED, FEATS, load, folds, host_path, save_preds, load_preds, write_sub

CHAR15 = dict(analyzer='char', ngram_range=(1, 5), min_df=2, sublinear_tf=True, max_features=500_000, lowercase=False)
WORD12 = dict(analyzer='word', token_pattern=r'[A-Za-z0-9]+', ngram_range=(1, 2), min_df=2, sublinear_tf=True)

# имя: (часть URL, параметры векторизаторов, классификатор)
MODELS = {
    'tfidf': ('full', [CHAR15, WORD12], 'lr10'),
    'host_char': ('host', [dict(analyzer='char_wb', ngram_range=(2, 6), min_df=2, sublinear_tf=True)], 'lr8'),
    'path_char': ('path', [dict(analyzer='char', ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=400_000)], 'lr8'),
    'full_sgd': ('full', [dict(analyzer='char', ngram_range=(1, 6), min_df=2, sublinear_tf=True, lowercase=False,
                               max_features=1_500_000)], 'sgd'),
}


def make_clf(kind):
    if kind == 'sgd':
        return SGDClassifier(loss='modified_huber', alpha=2e-6, max_iter=50, tol=None, random_state=SEED)
    return LogisticRegression(C=10 if kind == 'lr10' else 8, max_iter=3000, solver='liblinear')


def texts():
    tr, te, _ = load()
    allu = pd.concat([tr.url, te.url]).reset_index(drop=True)
    H = [host_path(u) for u in allu]
    return {'full': allu, 'host': pd.Series([h for h, _ in H]), 'path': pd.Series([r for _, r in H])}


def vectorize(vps, fit_on, *others, stack=False):
    """fit_transform на fit_on и transform на others.

    Порядок операций повторяет исходные скрипты буквально: SGD чувствителен к порядку
    ненулевых элементов в CSR, и даже hstack одной матрицы меняет его результат.
    """
    vs = [TfidfVectorizer(**vp) for vp in vps]
    join = lambda ms: sp.hstack(ms).tocsr() if (stack or len(ms) > 1) else ms[0].tocsr()
    return [join([v.fit_transform(fit_on) for v in vs])] + [join([v.transform(t) for v in vs]) for t in others]


def run_baseline(T, y, n):
    """Шаг 1 в исходном виде: векторизатор учится только на обучающей части фолда,
    модель для теста - на всём train. Сохраняем как было, чтобы не сдвигать результаты."""
    _, vps, kind = MODELS['tfidf']; full = T['full']; tr_txt, te_txt = full.iloc[:n], full.iloc[n:]
    oof = np.zeros(n)
    for i, j in folds(y):
        Xa, Xb = vectorize(vps, tr_txt.iloc[i], tr_txt.iloc[j])
        oof[j] = make_clf(kind).fit(Xa, y[i]).predict_proba(Xb)[:, 1]
    Xa, Xb = vectorize(vps, tr_txt, te_txt)
    pt = make_clf(kind).fit(Xa, y).predict_proba(Xb)[:, 1]
    return oof, pt


def run(name, T, y, n, pl_mask, pl_y, suffix):
    key, vps, kind = MODELS[name]
    if name == 'tfidf' and pl_mask is None:
        oof, pt = run_baseline(T, y, n)
    else:
        # словарь n-грамм строится по train и test вместе: меток здесь нет, это не утечка
        (X,) = vectorize(vps, T[key], stack=pl_mask is not None); Xtr, Xte = X[:n], X[n:]
        Xpl = Xte[pl_mask] if pl_mask is not None else None
        oof = np.zeros(n); pt = np.zeros(Xte.shape[0]); F = folds(y)
        for i, j in F:
            Xa, ya = (sp.vstack([Xtr[i], Xpl]), np.r_[y[i], pl_y]) if Xpl is not None else (Xtr[i], y[i])
            m = make_clf(kind).fit(Xa, ya)
            oof[j] = m.predict_proba(Xtr[j])[:, 1]; pt += m.predict_proba(Xte)[:, 1] / len(F)
    save_preds(name + suffix, oof, pt)
    return name + suffix, round(f1_score(y, oof > .5), 4), oof, pt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', nargs='+', default=list(MODELS), choices=list(MODELS))
    ap.add_argument('--pl', help='имя стека для псевдоразметки (artifacts/preds/test_<имя>.npy)')
    ap.add_argument('--thr', type=float, default=0.99, help='порог уверенности стека для псевдоразметки')
    ap.add_argument('--suffix', default='', help='суффикс имён результатов, например _pl')
    a = ap.parse_args()

    tr, te, y = load(); n = len(tr)
    pl_mask = pl_y = None
    if a.pl:
        E = pd.read_csv(FEATS / 'ext_feats.csv').iloc[n:].reset_index(drop=True)
        _, p = load_preds(a.pl); known = (E.x_lab_url >= 0).values
        pl_mask = known | (p > a.thr) | (p < 1 - a.thr)
        pl_y = np.where(known, (E.x_lab_url > .5).astype(int), (p > .5).astype(int))[pl_mask]
        print(f'pseudo-labelled test rows: {pl_mask.sum()}, pos rate {pl_y.mean():.3f}', flush=True)

    T = texts()
    res = Parallel(n_jobs=min(len(a.models), 4))(delayed(run)(m, T, y, n, pl_mask, pl_y, a.suffix) for m in a.models)
    for name, f1, oof, pt in res:
        print(name, 'OOF F1', f1, flush=True)
        if name == 'tfidf':   # первый сабмит исходной версии: порог подбирался по OOF
            ths = np.arange(.2, .8, .01); t = ths[int(np.argmax([f1_score(y, oof > t) for t in ths]))]
            write_sub('tfidf_lr', pt > t)


if __name__ == '__main__':
    main()
