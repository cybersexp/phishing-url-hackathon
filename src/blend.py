"""Шаг 8. Финальный файл ответов: среднее вероятностей нескольких стеков, порог 0.5.

  python src/blend.py blend3 stack9s stack9cat stack10
  python src/blend.py blend4 stack9s stack9cat stack10 stack11

Первый аргумент - имя результата, дальше имена стеков (artifacts/preds/oof|test_<имя>.npy).
Пишет submissions/sub_<имя>.csv. Если data/train.csv на месте, печатает OOF F1 смеси;
без него тоже работает - для быстрой проверки по приложенным вероятностям данные не нужны.
"""
import argparse, numpy as np
from common import DATA, load, load_preds, save_preds, write_sub


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('name'); ap.add_argument('parts', nargs='+')
    ap.add_argument('--thr', type=float, default=0.5)
    a = ap.parse_args()
    P = [load_preds(k) for k in a.parts]
    oof, p = np.mean([o for o, _ in P], 0), np.mean([t for _, t in P], 0)
    if (DATA / 'train.csv').exists():
        from sklearn.metrics import f1_score
        print(a.name, 'из', a.parts, 'OOF F1', round(f1_score(load()[2], oof > a.thr), 4))
    save_preds(a.name, oof, p); write_sub(a.name, p > a.thr)
    print(f'submissions/sub_{a.name}.csv: {len(p)} строк, доля фишинга {np.mean(p > a.thr):.4f}')


if __name__ == '__main__':
    main()
