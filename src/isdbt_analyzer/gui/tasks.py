"""Execução de tarefas em segundo plano com sinais Qt."""

from __future__ import annotations

import traceback

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from ..core.fec import ChainResult, simulate_chain


class _Signals(QObject):
    done = Signal(object, object)       # (token, resultado)
    failed = Signal(object, str)
    progress = Signal(int, str)


class Task(QRunnable):
    def __init__(self, token, fn, *args, with_progress=False, **kwargs) -> None:
        super().__init__()
        self.token = token
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self.signals = _Signals()
        if with_progress:
            self.kwargs["progress"] = lambda p, m: self.signals.progress.emit(int(p), str(m))

    def run(self) -> None:
        try:
            try:
                res = self.fn(*self.args, **self.kwargs)
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                self.signals.failed.emit(self.token, str(exc))
                return
            self.signals.done.emit(self.token, res)
        except RuntimeError:
            pass  # janela fechada enquanto a tarefa rodava
        finally:
            _alive.discard(self)


_alive: set = set()


def start(task: Task) -> Task:
    _alive.add(task)  # mantém a referência até o fim da execução
    task.setAutoDelete(False)
    QThreadPool.globalInstance().start(task)
    return task


def measure_point(modulation, code_rate, channel_state, cn_db, packets=768, rng=None):
    """Mede PER e taxa de erro de byte num ponto (para canais sem tabela)."""
    import numpy as np
    rng = rng or np.random.default_rng()
    acc = ChainResult()
    while acc.packets < packets:
        r, _ = simulate_chain(modulation, code_rate, channel_state, cn_db, 128, rng)
        acc.add(r)
    return acc.per, acc.byte_error_rate
