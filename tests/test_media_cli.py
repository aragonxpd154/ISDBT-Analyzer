import numpy as np
import pytest

from isdbt_analyzer import cli
from isdbt_analyzer.core import config_io, media
from isdbt_analyzer.core.channel import AWGN, ChannelState
from isdbt_analyzer.core.ofdm import TransmissionConfig


def test_image_high_snr_is_perfect():
    """Regressão da v1: mesmo a 60 dB a imagem voltava 94–99 % corrompida."""
    img = media.test_pattern(320, 180)
    st = ChannelState.build(AWGN, TransmissionConfig())
    for mod in ("QPSK", "16-QAM", "64-QAM"):
        out, stats = media.degrade_uncoded(img, mod, 45.0, st, luma_only=False, seed=1)
        assert stats["bit_errors"] == 0
        assert np.array_equal(np.asarray(out), np.asarray(img))


def test_image_low_snr_has_errors():
    img = media.test_pattern(160, 90)
    st = ChannelState.build(AWGN, TransmissionConfig())
    _, stats = media.degrade_uncoded(img, "64-QAM", 12.0, st, seed=2)
    assert stats["ber"] > 1e-2


def test_coded_image_packets():
    img = media.test_pattern(160, 90)
    out, st = media.degrade_coded(img, per=1.0, byte_error=0.1, luma_only=False, seed=3)
    assert st["pacotes_perdidos"] == st["pacotes"]
    out, st = media.degrade_coded(img, per=0.0, byte_error=0.1, luma_only=False, seed=3)
    assert st["bytes_errados"] == 0


def test_config_roundtrip(tmp_path):
    cfg = TransmissionConfig()
    p = config_io.save(tmp_path / "c.json", cfg, "Brasil A", 21.5)
    cfg2, ch, cn = config_io.load(p)
    assert cfg2 == cfg and ch == "Brasil A" and cn == 21.5


def test_parse_layer():
    lay = config_io.parse_layer("b:12:64qam:3/4:2")
    assert (lay.name, lay.segments, lay.modulation, lay.code_rate, lay.interleave) == ("B", 12, "64-QAM", "3/4", 2)
    with pytest.raises(ValueError):
        config_io.parse_layer("B:12")


def test_cli_rate(capsys):
    assert cli.main(["taxa", "-c", "A:1:QPSK:2/3:4", "-c", "B:12:64QAM:3/4:2"]) == 0
    out = capsys.readouterr().out
    assert "17.843" in out and "18.283" in out


def test_cli_performance(capsys):
    assert cli.main(["desempenho", "-c", "B:13:64QAM:3/4", "--cn", "20"]) == 0
    assert "margem" in capsys.readouterr().out


def test_cli_csv_and_report(tmp_path):
    assert cli.main(["curvas", "-o", str(tmp_path / "c.csv"), "--de", "10", "--ate", "12"]) == 0
    assert (tmp_path / "c.csv").read_text(encoding="utf-8").startswith("camada;")
    assert cli.main(["relatorio", "-o", str(tmp_path / "r.pdf"), "--cn", "20"]) == 0
    assert (tmp_path / "r.pdf").read_bytes()[:4] == b"%PDF"


def test_video_heavy_corruption_terminates(tmp_path):
    """Regressão: PTS corrompido fazia o FFmpeg gerar quadros sem fim e travar a simulação."""
    from isdbt_analyzer.core import video
    if not video.find_ffmpeg():
        pytest.skip("FFmpeg indisponível")
    r = video.simulate_video(None, 5e6, per=0.4, byte_error=0.05, workdir=tmp_path, seed=1, max_seconds=3)
    n_orig = video.count_frames(r.original_view)
    assert r.stats.corrupted > 0
    assert 0 < video.count_frames(r.corrupted_view) <= n_orig


def test_video_total_loss_shows_no_signal(tmp_path):
    """Regressão: com perda total, a PMT corrompida fazia o FFmpeg falhar ('no decoder found for avs2')."""
    from isdbt_analyzer.core import video
    if not video.find_ffmpeg():
        pytest.skip("FFmpeg indisponível")
    r = video.simulate_video(None, 5e6, per=1.0, byte_error=0.3, workdir=tmp_path, seed=1, max_seconds=2)
    assert r.no_signal
    assert video.count_frames(r.corrupted_view) == video.count_frames(r.original_view)
    assert 0 in r.stats.psi_pids and len(r.stats.psi_pids) >= 2   # PAT + PMT preservadas
