"""Шаги 4 и 7. Второй уровень: бустинг на признаках и вероятностях текстовых моделей.

Запуск:  python src/stack_lgb.py <имя> [флаги]
Имя задаёт файлы результата: artifacts/preds/oof|test_<имя>.npy, submissions/sub_<имя>.csv.

Стеки, вошедшие в финальные решения:
  stack9s   --ext --dns --rdap --seeds 5 --base tfidf_pl host_char_pl path_char_pl full_sgd_pl
  stack9cat --ext --dns --rdap --cat     --base tfidf_pl host_char_pl path_char_pl full_sgd_pl
  stack10   --ext --dns --rdap --knn     --base tfidf_pl host_char_pl path_char_pl full_sgd_pl
  stack11   --ext --dns --rdap           --base tfidf_pl2 host_char_pl2 path_char_pl2 full_sgd_pl2

Всегда добавляется out-of-fold кодирование зарегистрированного домена:
доля фишинга среди других строк train с тем же доменом (со сглаживанием).

--honest-es: в исходной версии ранняя остановка шла по тому же валидационному фолду,
на котором потом считается OOF F1, - число итераций подбиралось под оцениваемые строки,
и OOF-оценка получалась слегка оптимистичной. С этим флагом из обучающей части фолда
отрезается 10% под раннюю остановку, а валидационный фолд используется только для оценки.
Без флага поведение совпадает с исходным (для воспроизведения приложенных результатов).
"""
import argparse, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from common import SEED, FEATS, N_JOBS, load, folds, host, registered, save_preds, load_preds, write_sub
from features import build_v2

RDAP_REF = pd.Timestamp('2026-10-13')   # дата сбора RDAP, от неё считается возраст домена в днях
LGB = dict(objective='binary', learning_rate=0.02, num_leaves=31, min_child_samples=30, feature_fraction=0.6,
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=3.0, cat_smooth=20, verbose=-1, seed=SEED)


def concat_split(Xtr, Xte, F, n):
    return (pd.concat([Xtr, F.iloc[:n].reset_index(drop=True)], axis=1),
            pd.concat([Xte, F.iloc[n:].reset_index(drop=True)], axis=1))


def build_matrix(a):
    tr, te, y = load(); n = len(tr)
    Xtr, Xte, regs = build_v2(tr.url, te.url)   # лексика и бренды, из кэша при повторных запусках
    Xte['tld'] = pd.Categorical(Xte['tld'].astype(str), categories=Xtr['tld'].cat.categories)

    # вероятности первого уровня: для train это out-of-fold значения
    for b in a.base:
        Xtr['p_' + b], Xte['p_' + b] = load_preds(b)

    if a.ext:
        Xtr, Xte = concat_split(Xtr, Xte, pd.read_csv(FEATS / 'ext_feats.csv'), n)
    hosts = (tr.url.map(host), te.url.map(host)) if (a.dns or a.rdap) else None
    if a.dns:
        D = pd.read_csv(FEATS / 'dns_feats.csv').set_index('host')
        for X, h in zip((Xtr, Xte), hosts):
            for c in ['a_cnt', 'resolved', 'cname', 'mx']: X['dns_' + c] = h.map(D[c]).fillna(-3).values   # -3: хост не проверялся
    if a.rdap:
        Rd = pd.read_csv(FEATS / 'rdap_feats.csv').drop_duplicates('reg').set_index('reg')
        for c in ['created', 'expires', 'changed']:
            Rd[c + '_d'] = (RDAP_REF - pd.to_datetime(Rd[c], errors='coerce', utc=True).dt.tz_localize(None)).dt.days
        for X, h in zip((Xtr, Xte), hosts):
            r = h.map(registered)
            X['rdap_code'] = r.map(Rd['code']).fillna(-9).values   # -9: домен не запрашивался
            X['rdap_n_status'] = r.map(Rd['n_status']).fillna(-9).values
            for c in ['created_d', 'expires_d', 'changed_d']: X['rdap_' + c] = r.map(Rd[c]).values   # NaN, если дата неизвестна
    if a.knn:
        Xtr, Xte = concat_split(Xtr, Xte, pd.read_csv(FEATS / 'knn_feats.csv'), n)

    # out-of-fold кодирование домена: для валидационного фолда статистика считается по остальным фолдам
    F = folds(y); rtr, rte = regs.iloc[:n].reset_index(drop=True), regs.iloc[n:].reset_index(drop=True)
    prior, k = y.mean(), 5   # сглаживание к общей доле фишинга, вес 5 строк

    def stats(idx):
        return pd.DataFrame({'r': rtr.iloc[idx].values, 'y': y[idx]}).groupby('r').y.agg(['sum', 'count'])

    te_oof = np.full(n, np.nan); cnt_oof = np.zeros(n)
    for i, j in F:
        g = stats(i); s = rtr.iloc[j].map(g['sum']); c = rtr.iloc[j].map(g['count']).fillna(0)
        te_oof[j] = ((s.fillna(0) + prior * k) / (c + k)).values; cnt_oof[j] = c.values
    g = stats(np.arange(n)); c = rte.map(g['count']).fillna(0)
    Xtr['dom_te'], Xtr['dom_cnt'] = te_oof, cnt_oof
    Xte['dom_te'], Xte['dom_cnt'] = ((rte.map(g['sum']).fillna(0) + prior * k) / (c + k)).values, c.values
    return Xtr, Xte, y, F


def fit_fold(a, Xa, ya, Xs, ys, Xb, Xt, seed):
    """Обучение на (Xa, ya) с ранней остановкой по (Xs, ys); предсказания для Xb и Xt."""
    if a.cat:
        from catboost import CatBoostClassifier
        cats = [c for c in Xa.columns if str(Xa[c].dtype) == 'category']
        Xa, Xs, Xb, Xt = (X.astype({c: str for c in cats}) for X in (Xa, Xs, Xb, Xt))   # CatBoost ждёт категории строками
        m = CatBoostClassifier(iterations=5000, learning_rate=0.03, depth=7, l2_leaf_reg=3, random_seed=seed, thread_count=N_JOBS,
                               eval_metric='Logloss', od_type='Iter', od_wait=200, verbose=False, cat_features=cats, allow_writing_files=False)
        m.fit(Xa, ya, eval_set=(Xs, ys))
        return m.predict_proba(Xb)[:, 1], m.predict_proba(Xt)[:, 1], m.get_best_iteration(), None
    P = dict(LGB, seed=seed, n_jobs=N_JOBS)
    m = lgb.train(P, lgb.Dataset(Xa, ya), 5000, valid_sets=[lgb.Dataset(Xs, ys)], callbacks=[lgb.early_stopping(200, verbose=False)])
    return m.predict(Xb, num_iteration=m.best_iteration), m.predict(Xt, num_iteration=m.best_iteration), m.best_iteration, m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('name')
    ap.add_argument('--base', nargs='+', default=['tfidf', 'host_char', 'path_char', 'full_sgd'],
                    help='вероятности первого уровня (artifacts/preds/oof|test_<имя>.npy)')
    for f, h in [('ext', 'признаки сторонних источников'), ('dns', 'DNS-признаки хоста'), ('rdap', 'данные о регистрации домена'),
                 ('knn', 'признаки ближайших соседей'), ('cat', 'CatBoost вместо LightGBM'),
                 ('honest_es', 'ранняя остановка на отдельных 10%% обучающей части')]:
        ap.add_argument('--' + f.replace('_', '-'), action='store_true', help=h)
    ap.add_argument('--seeds', type=int, default=1, help='усреднить N запусков с разными seed (фолды не меняются)')
    a = ap.parse_args()

    Xtr, Xte, y, F = build_matrix(a)
    n = len(y); oof = np.zeros(n); pt = np.zeros(len(Xte)); m = None
    for sd in range(a.seeds):
        for f, (i, j) in enumerate(F):
            if a.honest_es:
                ia, ies = train_test_split(i, test_size=0.1, stratify=y[i], random_state=SEED + sd)
            else:
                ia, ies = i, j   # исходное поведение
            pj, ptf, bi, m = fit_fold(a, Xtr.iloc[ia], y[ia], Xtr.iloc[ies], y[ies], Xtr.iloc[j], Xte, SEED + sd)
            oof[j] += pj / a.seeds; pt += ptf / len(F) / a.seeds   # усреднение по фолдам и по seed
            print(sd, f, bi, round(f1_score(y[j], pj > .5), 4), flush=True)

    ths = np.arange(.2, .8, .01); fs = [f1_score(y, oof > t) for t in ths]; t = ths[int(np.argmax(fs))]
    print('OOF F1@0.5', round(f1_score(y, oof > .5), 4), '| best', round(max(fs), 4), 'thr', round(t, 2),
          '(порог подобран на тех же OOF - оценка оптимистична)')
    save_preds(a.name, oof, pt)
    write_sub(a.name, pt > .5)   # единый порог 0.5, как в финальной смеси
    if m is not None:
        print(pd.Series(m.feature_importance('gain'), Xtr.columns).sort_values(ascending=False).head(20).round(0))


if __name__ == '__main__':
    main()
