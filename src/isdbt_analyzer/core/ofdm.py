"""Configuração de transmissão (modo, guarda, camadas A/B/C) e grandezas derivadas.

A taxa líquida de uma camada é calculada exatamente pela estrutura de quadro:

    R_layer = Nseg · Ndados · k · Rc · (188/204) / Ts

onde Ndados = portadoras de dados por segmento, k = bits/símbolo,
Rc = taxa do código convolucional, Ts = Tu·(1+GI). O resultado bate com as
tabelas da ABNT NBR 15601 (ex.: 13 seg, 64-QAM, 3/4, GI 1/8 → 18,25 Mb/s).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from . import params as P


@dataclass(frozen=True)
class Layer:
    name: str = "B"
    segments: int = 12
    modulation: str = "64-QAM"
    code_rate: str = "3/4"
    interleave: int = 2  # valor de I (entrelaçamento temporal)

    def normalized(self) -> Layer:
        return replace(
            self,
            modulation=P.normalize_modulation(self.modulation),
            code_rate=P.normalize_code_rate(self.code_rate),
        )

    @property
    def bits_per_symbol(self) -> int:
        return P.MODULATIONS[P.normalize_modulation(self.modulation)]

    @property
    def rate_fraction(self):
        return P.CODE_RATES[P.normalize_code_rate(self.code_rate)]

    @property
    def label(self) -> str:
        return (f"Camada {self.name}: {self.segments} seg · {P.normalize_modulation(self.modulation)}"
                f" · {P.normalize_code_rate(self.code_rate)}")


@dataclass(frozen=True)
class TransmissionConfig:
    mode: int = 3
    guard: str = "1/16"
    layers: tuple[Layer, ...] = field(default_factory=lambda: (
        Layer("A", 1, "QPSK", "2/3", 4),
        Layer("B", 12, "64-QAM", "3/4", 2),
    ))

    # ---------------------------------------------------------------- validação
    def validate(self) -> list[str]:
        """Retorna avisos (lista vazia = configuração válida). Erros graves levantam ValueError."""
        if self.mode not in P.MODES:
            raise ValueError(f"Modo inválido: {self.mode}")
        P.normalize_guard(self.guard)
        warnings: list[str] = []
        total = 0
        names = set()
        for layer in self.layers:
            layer.normalized()
            if layer.name in names:
                raise ValueError(f"Camada {layer.name} repetida")
            names.add(layer.name)
            if layer.segments < 1:
                raise ValueError(f"Camada {layer.name}: precisa de pelo menos 1 segmento")
            if layer.interleave not in P.MODES[self.mode].interleave_lengths:
                raise ValueError(
                    f"Camada {layer.name}: I={layer.interleave} não existe no modo {self.mode} "
                    f"(valores: {P.MODES[self.mode].interleave_lengths})")
            total += layer.segments
        if total > P.N_SEGMENTS:
            raise ValueError(f"Total de segmentos = {total} (máximo {P.N_SEGMENTS})")
        if total < P.N_SEGMENTS:
            warnings.append(f"Apenas {total} de {P.N_SEGMENTS} segmentos em uso.")
        if len(self.layers) > 1 and self.layers[0].name == "A" and self.layers[0].segments != 1:
            warnings.append("Recepção parcial (one-seg) exige a camada A com 1 segmento.")
        return warnings

    # ---------------------------------------------------------------- tempos
    @property
    def mode_params(self) -> P.Mode:
        return P.MODES[self.mode]

    @property
    def guard_fraction(self) -> float:
        return float(P.GUARD_INTERVALS[P.normalize_guard(self.guard)])

    @property
    def tu(self) -> float:
        """Duração útil do símbolo (s)."""
        return self.mode_params.fft_size / P.SAMPLE_RATE_HZ

    @property
    def tg(self) -> float:
        """Duração do intervalo de guarda (s)."""
        return self.tu * self.guard_fraction

    @property
    def ts(self) -> float:
        """Duração total do símbolo OFDM (s)."""
        return self.tu + self.tg

    @property
    def carrier_spacing(self) -> float:
        return 1.0 / self.tu

    @property
    def occupied_bandwidth(self) -> float:
        return self.mode_params.total_carriers * self.carrier_spacing

    @property
    def frame_duration(self) -> float:
        return P.SYMBOLS_PER_FRAME * self.ts

    @property
    def sfn_max_distance_m(self) -> float:
        """Diferença máxima de percurso entre transmissores SFN coberta pela guarda."""
        return self.tg * P.C_LIGHT

    # ---------------------------------------------------------------- taxas
    def layer_rate(self, layer: Layer) -> float:
        """Taxa líquida de TS (bit/s) de uma camada."""
        layer = layer.normalized()
        bits_per_ofdm_symbol = (layer.segments * self.mode_params.data_carriers_per_seg
                                * layer.bits_per_symbol * float(layer.rate_fraction))
        return bits_per_ofdm_symbol * (P.RS_K / P.RS_N) / self.ts

    def layer_tsp_per_frame(self, layer: Layer) -> float:
        """Pacotes TS (204 bytes) transmitidos por quadro OFDM na camada."""
        layer = layer.normalized()
        return (layer.segments * self.mode_params.data_carriers_per_seg
                * layer.bits_per_symbol * float(layer.rate_fraction)) / 8

    def total_rate(self) -> float:
        return sum(self.layer_rate(lay) for lay in self.layers)

    def interleaver_delay(self, layer: Layer) -> float:
        """Atraso aproximado do entrelaçador temporal (s)."""
        return P.TIME_INTERLEAVER_DELAY_PER_I * layer.interleave * self.ts

    def summary(self) -> dict:
        return {
            "modo": self.mode,
            "guarda": self.guard,
            "Tu_us": self.tu * 1e6,
            "Tg_us": self.tg * 1e6,
            "Ts_us": self.ts * 1e6,
            "espacamento_Hz": self.carrier_spacing,
            "banda_ocupada_MHz": self.occupied_bandwidth / 1e6,
            "quadro_ms": self.frame_duration * 1e3,
            "sfn_km": self.sfn_max_distance_m / 1e3,
            "camadas": [
                {
                    "camada": lay.name,
                    "segmentos": lay.segments,
                    "modulacao": P.normalize_modulation(lay.modulation),
                    "codigo": P.normalize_code_rate(lay.code_rate),
                    "I": lay.interleave,
                    "taxa_Mbps": self.layer_rate(lay) / 1e6,
                    "tsp_por_quadro": self.layer_tsp_per_frame(lay),
                    "atraso_entrelacador_ms": self.interleaver_delay(lay) * 1e3,
                }
                for lay in self.layers
            ],
            "taxa_total_Mbps": self.total_rate() / 1e6,
        }


def full_seg_rate(modulation: str, code_rate, guard) -> float:
    """Taxa (Mb/s) com os 13 segmentos numa única camada — atalho compatível com a v1."""
    cfg = TransmissionConfig(mode=3, guard=P.normalize_guard(guard),
                             layers=(Layer("A", 13, modulation, P.normalize_code_rate(code_rate), 0),))
    return cfg.total_rate() / 1e6
