import math

import pytest

from isdbt_analyzer.core import params as P
from isdbt_analyzer.core.ofdm import Layer, TransmissionConfig, full_seg_rate


@pytest.mark.parametrize("mod,rate,gi,expected", [
    # Taxas de referência para 13 segmentos (ABNT NBR 15601 / ARIB STD-B31), Mb/s
    ("64-QAM", "3/4", "1/8", 18.255),
    ("64-QAM", "3/4", "1/16", 19.330),
    ("64-QAM", "7/8", "1/32", 23.234),
    ("QPSK", "1/2", "1/4", 3.651),
    ("16-QAM", "3/4", "1/8", 12.170),
])
def test_full_seg_rates(mod, rate, gi, expected):
    assert full_seg_rate(mod, rate, gi) == pytest.approx(expected, abs=0.002)


def test_rate_independent_of_mode():
    rates = {m: TransmissionConfig(m, "1/8", (Layer("A", 13, "64-QAM", "3/4", 0),)).total_rate() for m in (1, 2, 3)}
    assert rates[1] == pytest.approx(rates[2]) == pytest.approx(rates[3])


def test_typical_brazilian_config():
    cfg = TransmissionConfig()
    assert cfg.validate() == []
    a, b = cfg.layers
    assert cfg.layer_rate(a) / 1e6 == pytest.approx(0.4406, abs=1e-3)
    assert cfg.layer_rate(b) / 1e6 == pytest.approx(17.843, abs=1e-3)
    assert cfg.layer_tsp_per_frame(b) == 2592


def test_timing_mode3():
    cfg = TransmissionConfig(3, "1/16")
    assert cfg.tu * 1e6 == pytest.approx(1008.0)
    assert cfg.tg * 1e6 == pytest.approx(63.0)
    assert cfg.carrier_spacing == pytest.approx(992.063, abs=1e-3)
    assert cfg.occupied_bandwidth / 1e6 == pytest.approx(5.5724, abs=1e-4)
    assert cfg.sfn_max_distance_m / 1e3 == pytest.approx(18.89, abs=0.01)


def test_interleaver_delay_same_in_all_modes():
    d1 = TransmissionConfig(1, "1/8", (Layer("A", 13, "QPSK", "1/2", 16),))
    d3 = TransmissionConfig(3, "1/8", (Layer("A", 13, "QPSK", "1/2", 4),))
    assert d1.interleaver_delay(d1.layers[0]) == pytest.approx(d3.interleaver_delay(d3.layers[0]))


def test_validation_errors():
    with pytest.raises(ValueError):
        TransmissionConfig(3, "1/8", (Layer("A", 10, "QPSK", "1/2", 0), Layer("B", 5, "QPSK", "1/2", 0))).validate()
    with pytest.raises(ValueError):
        TransmissionConfig(3, "1/8", (Layer("A", 13, "QPSK", "1/2", 16),)).validate()


@pytest.mark.parametrize("raw,canon", [("qpsk", "QPSK"), ("16qam", "16-QAM"), ("QAM64", "64-QAM"), ("64–QAM", "64-QAM")])
def test_normalize_modulation(raw, canon):
    assert P.normalize_modulation(raw) == canon


def test_normalize_code_rate():
    assert P.normalize_code_rate(0.75) == "3/4"
    assert P.normalize_code_rate("7/8") == "7/8"
    assert not math.isnan(float(P.CODE_RATES["5/6"]))
