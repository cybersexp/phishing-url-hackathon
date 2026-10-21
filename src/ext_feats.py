"""Шаг 3. Признаки из сторонних источников (разрешены правилами хакатона).

Результат: artifacts/features/ext_feats.csv, строки идут в порядке train, затем test.

Источники (все публичные, скачиваются в папку ext/, ссылки в README):
  ext/phishdb/     - mitchellkrogza/Phishing.Database: списки фишинговых ссылок и доменов
  ext/phishtank/   - PhishTank online-valid.csv
  ext/openphish/   - OpenPhish feed.txt
  ext/phiusiil/    - UCI PhiUSIIL Phishing URL Dataset
  ext/tranco/      - Tranco, миллион самых популярных доменов
  ext/hf_ealv/     - HuggingFace ealvaradob/phishing-dataset, файл urls.json
  ext/hf_kmack/    - HuggingFace kmack/Phishing_urls, parquet
  ext/hf_piro/     - HuggingFace pirocheto/phishing-url, parquet
"""
import re, json, numpy as np, pandas as pd

from common import FEATS, EXT, EXTRACT as ext, load, host


def norm(u):
    # нормализация для точного сравнения: без схемы, без www, без хвостового слеша, в нижнем регистре
    return re.sub(r'^[a-z]+://', '', str(u).strip().lower()).removeprefix('www.').rstrip('/')


def host_of(u):
    return host(u).removeprefix('www.')


tr, te, _ = load()
urls = pd.concat([tr.url, te.url]).reset_index(drop=True)
N = urls.map(norm); H = urls.map(host_of); R = H.map(lambda h: ext(h).top_domain_under_public_suffix or h)


def lines(p):
    return [l.strip() for l in open(p, errors='ignore') if l.strip() and not l.startswith('#')]


# списки фишинговых ссылок и доменов
phish_links = {norm(l) for f in ['phishing-links-ACTIVE.txt', 'phishing-links-INACTIVE.txt'] for l in lines(EXT / 'phishdb' / f)}
pt = pd.read_csv(EXT / 'phishtank/online-valid.csv', usecols=['url'])
phish_links |= set(pt.url.map(norm)) | {norm(l) for l in lines(EXT / 'openphish/feed.txt')}
phish_hosts = {norm(l) for l in lines(EXT / 'phishdb/phishing-domains-ACTIVE.txt')}
phish_link_hosts = {l.split('/')[0] for l in phish_links}

# PhiUSIIL: в этом датасете label=1 означает легитимный сайт
pu = pd.read_csv(EXT / 'phiusiil/PhiUSIIL_Phishing_URL_Dataset.csv', usecols=['URL', 'label'])
pu_n = pu.URL.map(norm)
pu_phish, pu_legit = set(pu_n[pu.label == 0]), set(pu_n[pu.label == 1])
pu_legit_hosts = {l.split('/')[0] for l in pu_legit}

# рейтинг популярности Tranco: чем меньше ранг, тем известнее домен
tr_rank = pd.read_csv(EXT / 'tranco/top-1m.csv', header=None, names=['rank', 'dom']).set_index('dom')['rank']

F = pd.DataFrame({
    'x_link_phish': N.isin(phish_links).astype(int),
    'x_link_phish_noslash': N.map(lambda s: s in phish_links or s + '/' in phish_links).astype(int),
    'x_host_phishdom': H.isin(phish_hosts).astype(int),
    'x_host_in_phishlinks': H.isin(phish_link_hosts).astype(int),
    'x_pu_phish': N.isin(pu_phish).astype(int), 'x_pu_legit': N.isin(pu_legit).astype(int),
    'x_pu_legit_host': H.isin(pu_legit_hosts).astype(int),
    # логарифм ранга; домены вне миллиона получают условный ранг 2e6
    'x_tranco_reg': np.log1p(R.map(tr_rank).fillna(2e6)), 'x_tranco_host': np.log1p(H.map(tr_rank).fillna(2e6)),
})

y = tr.result.values; Ft = F.iloc[:len(tr)]
for c in F.columns:
    if c.startswith('x_tranco'):
        print(c, 'in top1m:', (Ft[c] < np.log1p(2e6)).mean().round(3), 'phish rate there', y[Ft[c] < np.log1p(1e5)].mean().round(3))
    else:
        print(c, 'train hits', Ft[c].sum(), 'phish rate', y[Ft[c] == 1].mean().round(3), 'test hits', F[c].iloc[len(tr):].sum())

# размеченные датасеты URL: точная метка по URL и доля фишинга по хосту
lab = [pd.DataFrame(json.load(open(EXT / 'hf_ealv/urls.json'))).rename(columns={'text': 'url'}),
       pd.concat([pd.read_parquet(f) for f in (EXT / 'hf_kmack').glob('*.parquet')]).rename(columns={'text': 'url'}),
       pd.concat([pd.read_parquet(f) for f in (EXT / 'hf_piro').glob('*.parquet')])[['url', 'status']]
         .assign(label=lambda d: (d.status == 'phishing').astype(int))[['url', 'label']],
       pu.rename(columns={'URL': 'url'}).assign(label=lambda d: 1 - d.label)]
L = pd.concat(lab, ignore_index=True)
L['n'] = L.url.map(norm); L['h'] = L.url.map(host_of)
url_lab = L.groupby('n').label.mean()
hg = L.groupby('h').label.agg(['mean', 'count'])
# -1 значит "в источниках не найдено"; бустинг обрабатывает это как отдельное значение
F2 = pd.DataFrame({'x_lab_url': N.map(url_lab).fillna(-1),
                   'x_lab_host_rate': H.map(hg['mean']).fillna(-1), 'x_lab_host_cnt': H.map(hg['count']).fillna(0)})
F = pd.concat([F, F2], axis=1); F.to_csv(FEATS / 'ext_feats.csv', index=False)

Ft = F2.iloc[:len(tr)]
print('x_lab_url known train', (Ft.x_lab_url >= 0).sum(), 'acc', ((Ft.x_lab_url > .5) == y)[Ft.x_lab_url >= 0].mean().round(4),
      'test known', (F2.x_lab_url.iloc[len(tr):] >= 0).sum())
print('x_lab_host known train', (Ft.x_lab_host_rate >= 0).sum(), 'test known', (F2.x_lab_host_rate.iloc[len(tr):] >= 0).sum())
