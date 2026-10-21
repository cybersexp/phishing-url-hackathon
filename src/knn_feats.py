"""Шаг 6в. Признаки ближайших соседей по символьным n-граммам URL (для стека stack10).

Для каждого URL ищутся 10 самых похожих URL из train по косинусному сходству
TF-IDF-векторов. Признаки: сходство и метка ближайшего соседа, взвешенная
метка топ-5 и топ-10, максимальное сходство с фишинговым и с безопасным соседом.

Чтобы не было утечки, для строк train соседи ищутся только среди других фолдов,
для строк test - среди всего train. Результат: artifacts/features/knn_feats.csv (train, затем test).
"""
import numpy as np, pandas as pd, time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score

from common import FEATS, N_JOBS, load, folds

K = 10
tr, te, y = load(); n = len(tr)
X = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), min_df=2, sublinear_tf=True, dtype=np.float32).fit_transform(pd.concat([tr.url, te.url]).str.lower())
Xtr, Xte = X[:n], X[n:]
COLS = ['knn_sim1', 'knn_lab1', 'knn_sim1_lab', 'knn_wmean5', 'knn_wmean10', 'knn_maxsim_phish', 'knn_maxsim_benign', 'knn_n_sim90', 'knn_mean_sim5']


def feats(Xq, Xref, yref):
    nn = NearestNeighbors(n_neighbors=K, metric='cosine', algorithm='brute', n_jobs=N_JOBS).fit(Xref)
    d, i = nn.kneighbors(Xq); s = 1 - d          # косинусное сходство
    lab = yref[i]; w = np.maximum(s, 0) + 1e-6   # веса соседей пропорциональны сходству
    return pd.DataFrame({'knn_sim1': s[:, 0], 'knn_lab1': lab[:, 0], 'knn_sim1_lab': s[:, 0] * (2 * lab[:, 0] - 1),
        'knn_wmean5': (w[:, :5] * lab[:, :5]).sum(1) / w[:, :5].sum(1), 'knn_wmean10': (w * lab).sum(1) / w.sum(1),
        'knn_maxsim_phish': np.where(lab == 1, s, 0).max(1), 'knn_maxsim_benign': np.where(lab == 0, s, 0).max(1),
        'knn_n_sim90': (s > 0.9).sum(1), 'knn_mean_sim5': s[:, :5].mean(1)})


t0 = time.time()
F = pd.DataFrame(index=range(n), columns=COLS, dtype=float)
# то же разбиение на фолды, что и во всех остальных скриптах
for f, (i, j) in enumerate(folds(y)):
    F.iloc[j] = feats(Xtr[j], Xtr[i], y[i]).values; print('fold', f, round(time.time() - t0), flush=True)
Ft = feats(Xte, Xtr, y); print('test', round(time.time() - t0), flush=True)
pd.concat([F, Ft], ignore_index=True).to_csv(FEATS / 'knn_feats.csv', index=False)
print('kNN-only F1 (wmean5>0.5):', round(f1_score(y, F.knn_wmean5.values > 0.5), 4), '| lab1:', round(f1_score(y, F.knn_lab1.values > 0.5), 4))
