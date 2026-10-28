import numpy as np
import pytest
from scipy.special import erfc

from isdbt_analyzer.core import modulation as M


def q(x):
    return 0.5 * erfc(x / np.sqrt(2))


@pytest.mark.parametrize("mod", ["QPSK", "16-QAM", "64-QAM"])
def test_unit_energy_and_gray(mod):
    pts, labels = M.constellation(mod)
    assert np.mean(np.abs(pts) ** 2) == pytest.approx(1.0)
    # vizinhos mais próximos diferem em exatamente 1 bit (Gray em 2D)
    d = np.abs(pts[:, None] - pts[None, :])
    dmin = np.min(d[d > 1e-9])
    for i in range(len(pts)):
        for j in np.flatnonzero(np.abs(d[i] - dmin) < 1e-9):
            assert bin(int(labels[i]) ^ int(labels[j])).count("1") == 1


@pytest.mark.parametrize("mod", ["QPSK", "16-QAM", "64-QAM"])
def test_noiseless_roundtrip(mod):
    rng = np.random.default_rng(0)
    k = {"QPSK": 2, "16-QAM": 4, "64-QAM": 6}[mod]
    bits = rng.integers(0, 2, 6000 * k, dtype=np.uint8)
    assert np.array_equal(M.demap_hard(M.map_bits(bits, mod), mod), bits)
    llr = M.llr_maxlog(M.map_bits(bits, mod), mod, 0.01)
    assert np.array_equal((llr < 0).astype(np.uint8), bits)


def test_qpsk_closed_form():
    eb = np.array([0.0, 4.0, 8.0])
    es = M.ebn0_to_esn0(eb, "QPSK")
    assert np.allclose(M.ber_uncoded("QPSK", es), q(np.sqrt(2 * 10 ** (eb / 10))), rtol=1e-9)


@pytest.mark.parametrize("mod,es", [("QPSK", 8.0), ("16-QAM", 14.0), ("64-QAM", 20.0)])
def test_exact_ber_matches_monte_carlo(mod, es):
    rng = np.random.default_rng(1)
    k = {"QPSK": 2, "16-QAM": 4, "64-QAM": 6}[mod]
    bits = rng.integers(0, 2, 300_000 * k, dtype=np.uint8)
    rx = M.awgn(M.map_bits(bits, mod), es, rng)
    mc = np.mean(M.demap_hard(rx, mod) != bits)
    th = M.ber_uncoded(mod, es)[0]
    assert mc == pytest.approx(th, rel=0.15)


def test_v1_bug_regression():
    """A v1 dava BER do QPSK 3 dB otimista; aqui Eb/N0 = 6 dB → 2,39e-3."""
    es = M.ebn0_to_esn0(6.0, "QPSK")
    assert M.ber_uncoded("QPSK", es)[0] == pytest.approx(2.388e-3, rel=1e-3)
