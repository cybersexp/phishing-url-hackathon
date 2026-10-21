"""Шаг 6б. Данные о регистрации домена через RDAP (замена WHOIS).

Для каждого зарегистрированного домена из train и test запрашиваются даты
регистрации, окончания и последнего изменения, а также число статусов.
Адрес RDAP-сервера для каждой зоны берётся из файла ext/rdap_bootstrap.json
(это https://data.iana.org/rdap/dns.json), чтобы не ходить через редиректор rdap.org.

Результат: artifacts/features/rdap_feats.csv с колонками reg, code, created, expires, changed, n_status.
  code 200 - данные получены, 404 - домен не зарегистрирован, -2 - ошибка сети.

Домены обрабатываются в случайном порядке с фиксированным seed и промежуточным
сохранением: если скрипт прервать, частичное покрытие будет одинаково случайным
для train и test. Снимок от 13.10.2026 приложен к пакету.
"""
import re, json, time, random, threading, pandas as pd, requests
from concurrent.futures import ThreadPoolExecutor

from common import FEATS, EXT, EXTRACT, load, host


tr, te, _ = load()
regs = sorted({EXTRACT(h).top_domain_under_public_suffix for h in pd.concat([tr.url, te.url]).map(host)} - {''})
regs = [r for r in regs if not re.fullmatch(r'[\d.]+', r)]   # IP-адреса пропускаем
random.Random(42).shuffle(regs)

# продолжение с места остановки, если есть частичный файл
PART = FEATS / 'rdap_feats.partial.csv'; done = pd.read_csv(PART) if PART.exists() else pd.DataFrame()
seen = set(done.reg) if not done.empty else set(); regs = [r for r in regs if r not in seen]
print('remaining', len(regs), flush=True)

# зона -> адрес RDAP-сервера реестра
BOOT = {}
for tlds, urls in (s[:2] for s in json.load(open(EXT / 'rdap_bootstrap.json'))['services']):
    for t in tlds: BOOT[t.lower()] = next((u for u in urls if u.startswith('https')), urls[0])

# по одной HTTP-сессии на поток
_local = threading.local()


def sess():
    if not hasattr(_local, 's'): _local.s = requests.Session(); _local.s.headers['Accept'] = 'application/rdap+json'
    return _local.s


def one(r):
    base = BOOT.get(r.rsplit('.', 1)[-1], 'https://rdap.org/')
    for a in range(2):
        try:
            x = sess().get(f'{base.rstrip("/")}/domain/{r}', timeout=15)
            if x.status_code == 429: time.sleep(3 + 3 * a); continue   # слишком много запросов, ждём и повторяем
            if x.status_code != 200: return (r, x.status_code, '', '', '', -1)
            j = x.json(); ev = {e.get('eventAction'): (e.get('eventDate') or '')[:10] for e in j.get('events', [])}
            return (r, 200, ev.get('registration', ''), ev.get('expiration', ''), ev.get('last changed', ''), len(j.get('status', [])))
        except Exception: time.sleep(1)
    return (r, -2, '', '', '', -1)


rows = []; t0 = time.time()


def flush():
    global done, rows
    done = pd.concat([done, pd.DataFrame(rows, columns=['reg', 'code', 'created', 'expires', 'changed', 'n_status'])], ignore_index=True)
    rows = []; done.to_csv(PART, index=False)


with ThreadPoolExecutor(32) as ex:
    for k, r in enumerate(ex.map(one, regs)):
        rows.append(r)
        if k % 500 == 0: flush(); print(k, round(time.time() - t0), flush=True)
flush(); done.to_csv(FEATS / 'rdap_feats.csv', index=False); print('done', round(time.time() - t0))
