"""Синтетические train/test и заглушки признаков для смоук-теста. Не реальные данные.

  python tests/make_synth.py <папка>/data <n_train> <n_test>
"""
import os, random, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from common import host, EXTRACT

r = random.Random(0); out = sys.argv[1]; NTR, NTE = int(sys.argv[2]), int(sys.argv[3])
good = ['google.com','wikipedia.org','github.com','bbc.co.uk','yandex.ru','amazon.com','apple.com','python.org','stackoverflow.com','nytimes.com','mail.ru','hse.ru']
words = ['news','docs','about','article','product','item','search','help','blog','user','2024','index','page']
bad_tld = ['xyz','top','tk','ru','info','online','site','click']
brands = ['paypal','apple','microsoft','amazon','netflix','sberbank','gosuslugi','dhl']
def benign():
    h = r.choice(good); h = (r.choice(['www.','','m.','en.']) + h)
    p = '/'.join(r.choice(words) for _ in range(r.randint(0,4)))
    return r.choice(['https://','http://','']) + h + ('/' + p if p else '') + (f'?id={r.randint(1,9999)}' if r.random()<.3 else '')
def phish():
    b = r.choice(brands); k = r.random()
    if k < .3: h = f'{b}-{r.choice(["secure","login","verify"])}.{r.choice(bad_tld)}'
    elif k < .5: h = f'{b}.com.{"".join(r.choice("abcdefghijk") for _ in range(8))}.{r.choice(bad_tld)}'
    elif k < .7: h = f'{r.randint(10,250)}.{r.randint(0,255)}.{r.randint(0,255)}.{r.randint(1,254)}'
    elif k < .85: h = f'{"".join(r.choice("abcdefghijklmnop0123456789") for _ in range(12))}.{r.choice(["weebly.com","web.app","000webhostapp.com"])}'
    else: h = b[:-1] + r.choice('aeiou') + b[-1] + '.com'
    p = r.choice(['/login.php','/signin/verify','/wp-includes/account/update.html','/secure/session?token=' + str(r.randint(1,10**9)),''])
    return r.choice(['http://','https://','']) + h + p
def mk(n):
    ys = [int(r.random() < .38) for _ in range(n)]; urls = [phish() if y else benign() for y in ys]; ys = [y ^ (r.random() < .03) for y in ys]; return pd.DataFrame({"url": urls, 'result': ys})
os.makedirs(out, exist_ok=True)
tr = mk(NTR); te = mk(NTE)
tr.to_csv(f'{out}/train.csv', index=False); te[['url']].to_csv(f'{out}/test.csv', index=False)
pd.DataFrame({'Id': range(NTE), 'Predicted': 0}).to_csv(f'{out}/sample_submission.csv', index=False)


# заглушки признаков сторонних источников, DNS и RDAP с теми же колонками, что в настоящих файлах
feats = Path(out).parent / 'artifacts' / 'features'; feats.mkdir(parents=True, exist_ok=True)
allu = pd.concat([tr.url, te.url]).reset_index(drop=True); g = np.random.default_rng(0); N = len(allu)
ylab = np.r_[tr.result.values, te.result.values]
known = g.random(N) < .3
pd.DataFrame({'x_link_phish': g.integers(0, 2, N), 'x_link_phish_noslash': g.integers(0, 2, N), 'x_host_phishdom': 0,
              'x_host_in_phishlinks': g.integers(0, 2, N), 'x_pu_phish': 0, 'x_pu_legit': 0, 'x_pu_legit_host': 0,
              'x_tranco_reg': g.uniform(5, 15, N), 'x_tranco_host': g.uniform(5, 15, N),
              'x_lab_url': np.where(known, ylab, -1.0), 'x_lab_host_rate': -1.0, 'x_lab_host_cnt': 0.0}).to_csv(feats / 'ext_feats.csv', index=False)
hs = sorted(set(allu.map(host)) - {''})
pd.DataFrame({'host': hs, 'a_cnt': g.integers(-2, 4, len(hs)), 'resolved': g.integers(0, 2, len(hs)), 'cname': 0, 'mx': 0}).to_csv(feats / 'dns_feats.csv', index=False)
rg = sorted({EXTRACT(h).top_domain_under_public_suffix for h in hs} - {''})
pd.DataFrame({'reg': rg, 'code': 200, 'created': '2015-01-01', 'expires': '2027-01-01', 'changed': '2026-01-01', 'n_status': 1}).to_csv(feats / 'rdap_feats.csv', index=False)
