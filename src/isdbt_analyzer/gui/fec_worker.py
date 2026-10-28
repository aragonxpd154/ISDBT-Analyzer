"""Processo separado que roda a cadeia Viterbi + RS sem travar a interface.

O laço do Viterbi é Python + NumPy. Numa thread, ele disputaria o GIL com a
interface; por isso roda num processo próprio, que recebe parâmetros por uma
fila e devolve contadores parciais por outra.
"""

from __future__ import annotations

import multiprocessing as mp
import queue

import numpy as np


def _loop(cmd_q, out_q) -> None:  # executa no processo filho
    from ..core.channel import ChannelState
    from ..core.fec import simulate_chain

    params = None
    state = None
    rng = np.random.default_rng()
    running = True
    while True:
        try:
            block = params is None or not running
            msg = cmd_q.get(timeout=None if block else 0.0)
        except queue.Empty:
            msg = None
        if msg is not None:
            kind = msg[0]
            if kind == "stop":
                return
            if kind == "pause":
                running = False
                continue
            if kind == "resume":
                running = True
                continue
            if kind == "params":
                params = msg[1]
                state = ChannelState.build(params["profile"], params["cfg"])
                running = True
                continue
        if params is None or not running:
            continue
        res, _ = simulate_chain(params["modulation"], params["code_rate"], state, params["cn"],
                                params.get("batch", 96), rng)
        out_q.put((params["epoch"], {
            "bits": res.bits, "raw_bits": res.raw_bits, "raw_bit_errors": res.raw_bit_errors,
            "vit_bit_errors": res.vit_bit_errors, "bytes_total": res.bytes_total,
            "vit_byte_errors": res.vit_byte_errors, "packets": res.packets,
            "rs_failures": res.rs_failures, "hist": res.byte_err_hist,
        }))


class FecWorker:
    def __init__(self) -> None:
        ctx = mp.get_context("spawn")
        self.cmd_q = ctx.Queue()
        self.out_q = ctx.Queue()
        self.proc = ctx.Process(target=_loop, args=(self.cmd_q, self.out_q), daemon=True,
                                name="isdbt-fec-worker")
        self.proc.start()
        self.epoch = 0

    def set_params(self, cfg, profile, modulation: str, code_rate: str, cn: float) -> int:
        self.epoch += 1
        # descarta resultados antigos ainda na fila
        self.drain()
        self.cmd_q.put(("params", {"epoch": self.epoch, "cfg": cfg, "profile": profile,
                                   "modulation": modulation, "code_rate": code_rate, "cn": cn}))
        return self.epoch

    def pause(self, paused: bool) -> None:
        self.cmd_q.put(("pause",) if paused else ("resume",))

    def drain(self) -> list:
        out = []
        while True:
            try:
                out.append(self.out_q.get_nowait())
            except queue.Empty:
                return out
            except (EOFError, OSError):
                return out

    def stop(self) -> None:
        try:
            self.cmd_q.put(("stop",))
            self.proc.join(timeout=1.5)
        except Exception:  # noqa: BLE001
            pass
        if self.proc.is_alive():
            self.proc.terminate()
