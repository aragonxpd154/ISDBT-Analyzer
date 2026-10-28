"""Salvar e abrir configurações em JSON."""

from __future__ import annotations

import json
from pathlib import Path

from .ofdm import Layer, TransmissionConfig


def to_dict(cfg: TransmissionConfig, channel: str = "AWGN", cn_db: float | None = None) -> dict:
    return {
        "formato": "isdbt-analyzer/1",
        "modo": cfg.mode,
        "guarda": cfg.guard,
        "canal": channel,
        "cn_db": cn_db,
        "camadas": [
            {"nome": lay.name, "segmentos": lay.segments, "modulacao": lay.modulation,
             "codigo": lay.code_rate, "I": lay.interleave}
            for lay in cfg.layers
        ],
    }


def from_dict(d: dict) -> tuple[TransmissionConfig, str, float | None]:
    layers = tuple(Layer(c["nome"], int(c["segmentos"]), c["modulacao"], c["codigo"], int(c["I"]))
                   for c in d["camadas"])
    cfg = TransmissionConfig(mode=int(d["modo"]), guard=str(d["guarda"]), layers=layers)
    cfg.validate()
    return cfg, d.get("canal", "AWGN"), d.get("cn_db")


def save(path: str | Path, cfg: TransmissionConfig, channel: str = "AWGN", cn_db: float | None = None) -> Path:
    p = Path(path)
    p.write_text(json.dumps(to_dict(cfg, channel, cn_db), indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def load(path: str | Path):
    return from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def parse_layer(spec: str) -> Layer:
    """'B:12:64QAM:3/4:2' → Layer (nome:segmentos:modulação:código[:I])."""
    parts = spec.split(":")
    if len(parts) not in (4, 5):
        raise ValueError(f"Camada inválida {spec!r}. Formato: NOME:SEG:MOD:CODIGO[:I]  ex.: B:12:64QAM:3/4:2")
    name, seg, mod, rate = parts[:4]
    inter = int(parts[4]) if len(parts) == 5 else 0
    return Layer(name.upper(), int(seg), mod, rate, inter).normalized()
