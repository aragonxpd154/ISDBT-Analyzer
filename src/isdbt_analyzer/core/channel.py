"""Modelos de canal: AWGN, perfis de multipercurso Brasil A–E e eco SFN.

O canal é estático e aplicado no domínio da frequência: cada portadora k
recebe o ganho H(f_k) = Σ a_i·exp(-j2π f_k τ_i), com potência total
normalizada (Σ|a_i|² = 1). Com entrelaçamento em frequência ideal, cada
símbolo QAM cai numa portadora sorteada.

Ecos com atraso maior que o intervalo de guarda geram interferência
entre símbolos. O modelo aproxima isso convertendo a fração
(τ − Tg)/Tu da potência do eco em ruído adicional.

Os perfis Brasil A–E vêm da literatura dos testes de campo do SBTVD.
Confira com a norma que você usa: há publicações com pequenas
divergências (ex.: o 2º percurso do Brasil B aparece como −12 dB e como −1,2 dB).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ofdm import TransmissionConfig


@dataclass(frozen=True)
class ChannelProfile:
    name: str
    description: str
    delays_us: tuple[float, ...]
    atten_db: tuple[float, ...]

    @property
    def is_awgn(self) -> bool:
        return len(self.delays_us) <= 1

    def taps(self) -> tuple[np.ndarray, np.ndarray]:
        """(atrasos em s, amplitudes complexas normalizadas)."""
        d = np.asarray(self.delays_us, dtype=float) * 1e-6
        a = 10 ** (-np.asarray(self.atten_db, dtype=float) / 20)
        a = a / np.sqrt(np.sum(a ** 2))
        return d, a.astype(complex)

    def max_delay_us(self) -> float:
        return max(self.delays_us) - min(self.delays_us) if self.delays_us else 0.0


AWGN = ChannelProfile("AWGN", "Somente ruído gaussiano (sem multipercurso)", (0.0,), (0.0,))

PROFILES: dict[str, ChannelProfile] = {
    "AWGN": AWGN,
    "Brasil A": ChannelProfile(
        "Brasil A", "Recepção externa, ecos fracos",
        (0.0, 0.15, 2.22, 3.05, 5.86, 5.93), (0.0, 13.8, 16.2, 14.9, 13.6, 16.4)),
    "Brasil B": ChannelProfile(
        "Brasil B", "Recepção interna, ecos moderados",
        (0.0, 0.3, 3.5, 4.4, 9.5, 12.7), (0.0, 12.0, 4.0, 7.0, 15.0, 22.0)),
    "Brasil C": ChannelProfile(
        "Brasil C", "Recepção externa, ecos fortes próximos",
        (0.0, 0.089, 0.419, 1.506, 2.322, 2.799), (2.8, 0.0, 3.8, 0.1, 2.6, 1.3)),
    "Brasil D": ChannelProfile(
        "Brasil D", "Recepção externa com antena ruim, ecos fortes",
        (0.15, 0.63, 2.22, 3.05, 5.86, 5.93), (0.1, 3.8, 2.6, 1.3, 0.0, 2.8)),
    "Brasil E": ChannelProfile(
        "Brasil E", "Rede SFN: três sinais de mesma potência",
        (0.0, 1.0, 2.0), (0.0, 0.0, 0.0)),
}


def sfn_echo(delay_us: float, level_db: float = 0.0) -> ChannelProfile:
    """Dois transmissores SFN: sinal direto + eco com atraso e nível relativos."""
    return ChannelProfile(
        "Eco SFN", f"Eco de {level_db:+.1f} dB com {delay_us:.1f} µs",
        (0.0, float(delay_us)), (0.0, -float(level_db)))


def get_profile(name: str) -> ChannelProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"Perfil de canal desconhecido: {name!r}. Opções: {', '.join(PROFILES)}") from exc


def carrier_frequencies(cfg: TransmissionConfig) -> np.ndarray:
    """Frequências (Hz, relativas ao centro) das portadoras ativas."""
    n = cfg.mode_params.total_carriers
    return (np.arange(n) - (n - 1) / 2) * cfg.carrier_spacing


def frequency_response(profile: ChannelProfile, cfg: TransmissionConfig) -> np.ndarray:
    """H(f) complexa em todas as portadoras ativas."""
    f = carrier_frequencies(cfg)
    d, a = profile.taps()
    d = d - d.min()
    return (a[None, :] * np.exp(-2j * np.pi * f[:, None] * d[None, :])).sum(axis=1)


def isi_noise_fraction(profile: ChannelProfile, cfg: TransmissionConfig) -> float:
    """Fração da potência recebida que vira interferência (ecos além da guarda)."""
    d, a = profile.taps()
    d = d - d.min()
    excess = np.clip((d - cfg.tg) / cfg.tu, 0.0, 1.0)
    return float(np.sum(np.abs(a) ** 2 * excess))


@dataclass
class ChannelState:
    """Canal pronto para simular: resposta por portadora + ISI equivalente."""
    profile: ChannelProfile
    h: np.ndarray            # H por portadora
    isi_fraction: float

    @classmethod
    def build(cls, profile: ChannelProfile, cfg: TransmissionConfig) -> ChannelState:
        if profile.is_awgn:
            return cls(profile, np.ones(1, dtype=complex), 0.0)
        h = frequency_response(profile, cfg)
        # C/N é medido no receptor: normaliza a potência média recebida na banda para 1
        h = h / np.sqrt(np.mean(np.abs(h) ** 2))
        return cls(profile, h, isi_noise_fraction(profile, cfg))

    def effective_esn0_db(self, esn0_db: float) -> float:
        """Es/N0 efetivo incluindo a ISI como ruído extra."""
        if self.isi_fraction <= 0:
            return esn0_db
        es = 10 ** (esn0_db / 10)
        useful = 1.0 - self.isi_fraction
        sinr = useful / (1.0 / es + self.isi_fraction)
        return float(10 * np.log10(max(sinr, 1e-12)))

    def power_gains(self) -> np.ndarray:
        return np.abs(self.h) ** 2

    def apply(self, symbols: np.ndarray, esn0_db: float, rng: np.random.Generator):
        """Passa símbolos pelo canal.

        Retorna (recebido_bruto, equalizado, variância_do_ruído_após_equalizar, h_usado).
        """
        n = symbols.size
        eff = self.effective_esn0_db(esn0_db)
        n0 = 10 ** (-eff / 10)
        if self.h.size == 1:
            h = np.ones(n, dtype=complex)
        else:
            h = self.h[rng.integers(0, self.h.size, n)]   # entrelaçamento em frequência ideal
        noise = np.sqrt(n0 / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        rx = h * symbols + noise
        hs = np.where(np.abs(h) < 1e-6, 1e-6, h)
        eq = rx / hs
        nv = n0 / np.abs(hs) ** 2
        return rx, eq, nv, h
