"""Точка входа в проект (как в типичном ML-репозитории).

Примеры:
  python main.py              # основной пайплайн (модели + препроцессинг)
  python main.py advanced     # GEE, психометрика, leaky integrator, калибровка
  python main.py all          # оба подряд
"""

from __future__ import annotations

import sys


def run_evidence() -> None:
    from scripts.evidence_integration_analysis import main

    main()


def run_advanced() -> None:
    from scripts.advanced_analysis import main

    main()


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'evidence'
    if cmd in ('evidence', 'train', 'pipeline'):
        run_evidence()
    elif cmd == 'advanced':
        run_advanced()
    elif cmd == 'all':
        run_evidence()
        run_advanced()
    else:
        print('Использование: python main.py [evidence | advanced | all]')
        sys.exit(1)
