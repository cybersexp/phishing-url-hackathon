"""Лексические признаки URL, считаются только по самой строке адреса.

build()    - около 90 признаков: длины, счётчики символов, энтропия, ключевые слова,
             бесплатные хостинги, сокращатели ссылок, IP вместо домена, punycode и т.п.
build_v2() - то же плюс признаки брендов (тайпсквоттинг, бренд в поддомене или пути)
             и частота домена/хоста по train и test вместе.
Модуль импортируется из stack_lgb.py. Результат build_v2 кэшируется на диск.
"""
import hashlib, math, re
from collections import Counter
from functools import lru_cache
from urllib.parse import urlsplit
import pandas as pd

from common import CACHE, EXTRACT as _ext, with_scheme, is_ip

# слова, которые часто встречаются в фишинговых адресах
KW = ['login', 'signin', 'sign-in', 'logon', 'verify', 'verif', 'account', 'update', 'secure', 'security', 'bank', 'confirm',
      'password', 'webscr', 'paypal', 'apple', 'microsoft', 'office', 'outlook', 'amazon', 'netflix', 'wallet', 'support',
      'billing', 'invoice', 'unlock', 'suspend', 'auth', 'wp-', 'admin', 'includes', 'index.php', '.php', '.html', 'free',
      'bonus', 'lucky', 'gift', 'ebay', 'facebook', 'instagram', 'whatsapp', 'dhl', 'post', 'recover', 'session', 'token', 'mail']
# бесплатные хостинги и конструкторы сайтов, где фишинг размещают чаще всего
FREE_HOSTS = ['000webhost', 'weebly', 'web.app', 'firebaseapp', 'wix', 'blogspot', 'github.io', 'netlify', 'vercel', 'glitch',
              'herokuapp', 'repl.co', 'ngrok', 'duckdns', 'sharepoint', 'godaddysites', 'webcindario', 'workers.dev', 'pages.dev',
              'square.site', 'ipfs', 'r2.dev', 'firebasestorage', 'appspot', 'azurewebsites', 'cloudfront', 'mybluehost', 'flywheelsites']
# сокращатели ссылок
SHORT = ['bit.ly', 'tinyurl', 'goo.gl', 't.co', 'ow.ly', 'is.gd', 'cutt.ly', 'rb.gy', 'shorturl', 'tiny.cc', 'qrco.de', 'linktr.ee']


def entropy(s):
    # энтропия Шеннона по символам: случайные строки вроде хэшей дают высокое значение
    if not s: return 0.0
    c = Counter(s); n = len(s)
    return -sum(v / n * math.log2(v / n) for v in c.values())


_RE_TOK = re.compile(r'[^A-Za-z0-9]')
_RE_DIGITS = re.compile(r'\d+')


def _safe_port(sp):
    # urlsplit бросает ValueError на некорректном порте, такие адреса тоже бывают
    try: sp.port; return True
    except ValueError: return False


def parse(url):
    """Один разбор URL на строку: (исходная строка, SplitResult, host, tldextract)."""
    u = url.strip()
    sp = urlsplit(with_scheme(u))
    try: host = (sp.hostname or '').lower()
    except ValueError: host = ''
    return u, sp, host, _ext(host)


def feats(u, sp, host, e):
    has_scheme = with_scheme(u) == u
    path, query = sp.path, sp.query
    dom, sub, suf = e.domain, e.subdomain, e.suffix   # домен второго уровня, поддомен, публичный суффикс
    low = u.lower()
    n_digits = sum(c.isdigit() for c in u)
    d = {
        'len': len(u), 'host_len': len(host), 'path_len': len(path), 'query_len': len(query), 'frag_len': len(sp.fragment),
        'https': int(low.startswith('https')), 'has_scheme': int(has_scheme), 'www': int(host.startswith('www.')),
        'n_dots_host': host.count('.'), 'n_hyph_host': host.count('-'), 'n_digit_host': sum(c.isdigit() for c in host),
        'n_sub': len(sub.split('.')) if sub else 0, 'sub_len': len(sub), 'dom_len': len(dom), 'suf_len': len(suf),
        'is_ip': int(is_ip(host)), 'port': int(sp.port is not None) if _safe_port(sp) else 1,
        'n_slash': path.count('/'), 'n_qparams': query.count('&') + (1 if query else 0), 'n_eq': u.count('='),
        'n_at': u.count('@'), 'n_pct': u.count('%'), 'n_tilde': u.count('~'), 'n_dslash': u[8:].count('//'),
        'n_digits': n_digits, 'n_upper': sum(c.isupper() for c in u), 'non_ascii': sum(ord(c) > 127 for c in u),
        'punycode': int('xn--' in host), 'ent_url': entropy(u), 'ent_host': entropy(host), 'ent_dom': entropy(dom),
        'digit_ratio': n_digits / max(len(u), 1), 'trailing_slash': int(u.endswith('/')),
        'path_depth': len([p for p in path.split('/') if p]), 'ends_ellipsis': int(u.endswith('...')),
        'free_host': int(any(h in host for h in FREE_HOSTS)), 'shortener': int(any(h in host for h in SHORT)),
        'http_in_path': int('http' in (path + query).lower()), 'amp_esc': int('&amp;' in u),
        'longest_tok': max((len(t) for t in _RE_TOK.split(u)), default=0),
        'max_dom_digits_run': max((len(t) for t in _RE_DIGITS.findall(dom)), default=0),
        'ext_php': int(path.lower().endswith('.php')), 'ext_html': int(path.lower().endswith(('.html', '.htm'))),
    }
    kw = [int(k in low) for k in KW]
    d['kw_count'] = sum(kw)
    for k, v in zip(KW, kw): d['kw_' + k] = v   # по одному бинарному признаку на ключевое слово
    return d


def build(urls):
    """Только лексические признаки (без брендов и частот)."""
    P = [parse(u) for u in urls]
    X = pd.DataFrame([feats(*p) for p in P])
    X['tld'] = pd.Categorical([p[3].suffix for p in P])   # публичный суффикс как категориальный признак
    return X, pd.Series([p[3].domain for p in P])


# бренды, под которые чаще всего маскируется фишинг
BRANDS = ['paypal', 'apple', 'icloud', 'microsoft', 'office365', 'outlook', 'live', 'hotmail', 'amazon', 'netflix', 'facebook',
          'instagram', 'whatsapp', 'google', 'gmail', 'yahoo', 'ebay', 'chase', 'wellsfargo', 'bankofamerica', 'citibank', 'hsbc',
          'barclays', 'lloyds', 'santander', 'natwest', 'halifax', 'dhl', 'fedex', 'usps', 'ups', 'dropbox', 'docusign', 'adobe',
          'linkedin', 'twitter', 'steam', 'steamcommunity', 'roblox', 'binance', 'coinbase', 'metamask', 'blockchain', 'americanexpress',
          'amex', 'visa', 'mastercard', 'sberbank', 'tinkoff', 'alibaba', 'aliexpress', 'rakuten', 'mercari', 'aeon', 'smbc', 'mufg',
          'orange', 'ovh', 'att', 'verizon', 'comcast', 'xfinity', 'irs', 'gov', 'telegram', 'discord', 'spotify', 'wetransfer', 'onedrive',
          'sharepoint', 'yandex', 'mail', 'vk', 'avito', 'ozon', 'wildberries', 'gosuslugi', 'postbank', 'credit-agricole', 'itau',
          'bradesco', 'caixa', 'nubank', 'davivienda', 'bancolombia', 'interac', 'desjardins', 'tangerine', 'rbc', 'scotiabank']


@lru_cache(maxsize=None)
def lev(a, b):
    # расстояние Левенштейна; если длины отличаются больше чем на 2, считать не нужно
    if a == b: return 0
    if abs(len(a) - len(b)) > 2: return 9
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


BRANDS4 = [b for b in BRANDS if len(b) >= 4]
BRANDS_SET = set(BRANDS)


@lru_cache(maxsize=None)
def _dom_brand(d):
    # всё, что зависит только от домена: доменов в разы меньше, чем URL
    md = min((lev(d, b) for b in BRANDS4), default=9)
    return int(d in BRANDS_SET), int(any(b in d and b != d for b in BRANDS4)), md


def brand_feats(url, dom, sub):
    low = url.lower(); d = dom.lower(); s = sub.lower()
    in_url = [b for b in BRANDS if b in low]
    is_dom, in_dom, md = _dom_brand(d)
    return {'brand_any': len(in_url),                       # сколько брендов упомянуто где угодно в URL
            'brand_is_domain': is_dom,                      # сам домен и есть бренд (скорее всего, легитимный сайт)
            'brand_not_domain': int(bool(in_url) and not is_dom),  # бренд есть, но домен чужой
            'brand_in_sub': int(any(b in s for b in BRANDS4)),     # бренд в поддомене
            'brand_in_dom_substr': in_dom,                  # бренд как часть домена
            'brand_lev_min': md, 'brand_typo': int(0 < md <= 2)}   # опечатка в 1-2 символа от бренда


def _build_v2(allu):
    P = [parse(u) for u in allu]
    X = pd.DataFrame([feats(*p) for p in P])
    X['tld'] = pd.Categorical([p[3].suffix for p in P])
    B = pd.DataFrame([brand_feats(p[0], p[3].domain, p[3].subdomain) for p in P])
    X = pd.concat([X, B], axis=1)
    hosts = pd.Series([p[2] for p in P])
    regs = pd.Series([p[3].top_domain_under_public_suffix or p[2] for p in P])
    # сколько раз домен и хост встречаются в train и test вместе (меток не требует)
    X['reg_freq'] = regs.map(regs.value_counts()).values
    X['host_freq'] = hosts.map(hosts.value_counts()).values
    X['_reg'] = regs.values
    return X


def build_v2(urls_train, urls_test, cache=True):
    """Признаки для train и test: лексика, бренды, частота домена и хоста по обеим выборкам.

    Результат кэшируется в artifacts/cache/ по хэшу списка URL: пять запусков стека
    в run_all.sh больше не пересчитывают одно и то же.
    """
    allu = pd.concat([urls_train, urls_test]).reset_index(drop=True)
    key = hashlib.sha1('\n'.join(allu.astype(str)).encode()).hexdigest()[:16]
    path = CACHE / f'lex_{key}.parquet'
    if cache and path.exists():
        X = pd.read_parquet(path)
    else:
        X = _build_v2(allu)
        if cache: X.to_parquet(path)
    regs = X.pop('_reg').reset_index(drop=True)
    n = len(urls_train)
    return X.iloc[:n].reset_index(drop=True), X.iloc[n:].reset_index(drop=True), regs
