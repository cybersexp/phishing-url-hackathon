"""Шаг 6а. Живые DNS-проверки для каждого уникального хоста из train и test.

Для хоста запрашиваются записи A, CNAME и MX. Результат: artifacts/features/dns_feats.csv
с колонками host, a_cnt, resolved, cname, mx.
  a_cnt = -1 означает NXDOMAIN (домена нет), -2 - ошибка или таймаут.

Состояние DNS со временем меняется, поэтому снимок от 13.10.2026 приложен
к пакету. Для точного воспроизведения результата скрипт запускать не нужно.
"""
import time, pandas as pd
from concurrent.futures import ThreadPoolExecutor
import dns.resolver

from common import FEATS, load, host, is_ip


tr, te, _ = load()
hosts = sorted(set(pd.concat([tr.url, te.url]).map(host)) - {''})
print('unique hosts', len(hosts), flush=True)

# используем системный резолвер
R = dns.resolver.Resolver(); R.timeout = 3; R.lifetime = 6


def q(h, t):
    # число записей заданного типа; отрицательные значения кодируют ошибки
    try: return len(R.resolve(h, t, raise_on_no_answer=False).rrset or [])
    except dns.resolver.NXDOMAIN: return -1
    except Exception: return -2


def one(h):
    # IP-адрес резолвить не нужно, считаем его "живым"
    if is_ip(h): return (h, 1, 1, 0, 0)
    a = q(h, 'A'); c = q(h, 'CNAME') if a != -1 else -1; m = q(h, 'MX') if a != -1 else -1
    return (h, a, int(a > 0), int(c > 0), int(m > 0))


t0 = time.time(); rows = []
with ThreadPoolExecutor(64) as ex:
    for k, r in enumerate(ex.map(one, hosts)):
        rows.append(r)
        if k % 5000 == 0: print(k, round(time.time() - t0), flush=True)
pd.DataFrame(rows, columns=['host', 'a_cnt', 'resolved', 'cname', 'mx']).to_csv(FEATS / 'dns_feats.csv', index=False)
print('done', round(time.time() - t0))
