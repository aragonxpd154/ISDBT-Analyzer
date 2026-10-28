import numpy as np
import pytest

from isdbt_analyzer.core import fec
from isdbt_analyzer.core.channel import AWGN, PROFILES, ChannelState, isi_noise_fraction, sfn_echo
from isdbt_analyzer.core.ofdm import TransmissionConfig
from isdbt_analyzer.core.tables import get_curve

CFG = TransmissionConfig()
AWGN_STATE = ChannelState.build(AWGN, CFG)

# C/N para BER 2e-4 após Viterbi (canal gaussiano, estimação perfeita) — valores de referência
# da literatura DVB-T/ISDB-T, que usam o mesmo código interno e os mesmos mapeamentos.
REFERENCE_QEF = {
    ("QPSK", "1/2"): 3.1, ("QPSK", "2/3"): 4.9, ("QPSK", "3/4"): 5.9, ("QPSK", "5/6"): 6.9, ("QPSK", "7/8"): 7.7,
    ("16-QAM", "1/2"): 8.8, ("16-QAM", "2/3"): 11.1, ("16-QAM", "3/4"): 12.5, ("16-QAM", "5/6"): 13.5,
    ("16-QAM", "7/8"): 13.9,
    ("64-QAM", "1/2"): 14.4, ("64-QAM", "2/3"): 16.5, ("64-QAM", "3/4"): 18.0, ("64-QAM", "5/6"): 19.3,
    ("64-QAM", "7/8"): 20.1,
}


@pytest.mark.parametrize("rate", ["1/2", "2/3", "3/4", "5/6", "7/8"])
def test_viterbi_noiseless(rate):
    rng = np.random.default_rng(3)
    res, _ = fec.simulate_chain("16-QAM", rate, AWGN_STATE, 60.0, 8, rng)
    assert res.vit_bit_errors == 0 and res.rs_failures == 0


@pytest.mark.parametrize("rate,r", [("1/2", 1 / 2), ("2/3", 2 / 3), ("3/4", 3 / 4), ("5/6", 5 / 6), ("7/8", 7 / 8)])
def test_puncture_ratio(rate, r):
    n_info = 840  # múltiplo de todos os períodos de puncionamento
    assert fec.puncture_mask(rate, n_info).sum() == pytest.approx(n_info / r)


@pytest.mark.parametrize("key,ref", REFERENCE_QEF.items())
def test_qef_matches_reference(key, ref):
    curve = get_curve(key[0], key[1], "AWGN")
    assert curve.required_esn0() == pytest.approx(ref, abs=0.75)


def test_coding_gain_is_real():
    """No QEF o Viterbi entrega BER 2e-4 onde o sinal sem FEC ainda tem BER alta."""
    from isdbt_analyzer.core.modulation import ber_uncoded
    c = get_curve("64-QAM", "3/4", "AWGN")
    q = c.required_esn0()
    assert ber_uncoded("64-QAM", q)[0] > 1e-2


def test_rs_cliff():
    p = np.array([1e-4, 1e-3, 1e-2, 5e-2])
    per = fec.rs_packet_error_rate(p)
    assert per[0] < 1e-15 and per[-1] > 0.5
    assert np.all(np.diff(per) > 0)


def test_monte_carlo_point_close_to_table():
    rng = np.random.default_rng(11)
    acc = fec.ChainResult()
    for _ in range(4):
        r, _ = fec.simulate_chain("QPSK", "2/3", AWGN_STATE, 4.5, 256, rng)
        acc.add(r)
    tab = get_curve("QPSK", "2/3", "AWGN").ber(4.5)[0]
    assert acc.vit_ber == pytest.approx(tab, rel=0.5)


def test_multipath_profiles_normalized():
    for prof in PROFILES.values():
        st = ChannelState.build(prof, CFG)
        assert np.mean(st.power_gains()) == pytest.approx(1.0, rel=0.05)


def test_echo_beyond_guard_creates_isi():
    assert isi_noise_fraction(sfn_echo(20.0, 0.0), CFG) == 0.0
    assert isi_noise_fraction(sfn_echo(200.0, 0.0), CFG) > 0.0
