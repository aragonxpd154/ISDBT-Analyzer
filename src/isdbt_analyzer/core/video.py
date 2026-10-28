"""Simulação de vídeo no nível de Transport Stream.

1. O vídeo é codificado em H.264 + AAC dentro de um MPEG-TS, com taxa
   compatível com a camada escolhida.
2. Cada pacote de 188 bytes falha com a PER prevista após o RS(204,188).
   O pacote que falha recebe o bit TEI (transport_error_indicator) e tem o
   payload corrompido, como acontece num receptor real.
3. Os dois TS (original e corrompido) são decodificados pelo FFmpeg com taxa
   de quadros constante, para tocarem lado a lado sincronizados.

O resultado mostra os artefatos reais da TV digital (macroblocos, congelamento
e cortes de áudio), que não aparecem numa simples adição de ruído aos pixels.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import params as P

Progress = Callable[[int, str], None]


def find_ffmpeg() -> str | None:
    """FFmpeg do sistema ou, na falta dele, o binário embutido do imageio-ffmpeg."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return None


@dataclass
class TsStats:
    packets: int = 0
    corrupted: int = 0
    per_target: float = 0.0
    by_pid: dict | None = None

    @property
    def per(self) -> float:
        return self.corrupted / self.packets if self.packets else 0.0


@dataclass
class VideoSimResult:
    original_ts: Path
    corrupted_ts: Path
    original_view: Path
    corrupted_view: Path
    stats: TsStats
    fps: float
    video_bitrate: float


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-8:])
        raise RuntimeError(f"FFmpeg falhou:\n{tail}")


def make_test_video(ffmpeg: str, out: Path, seconds: int = 10) -> Path:
    """Gera um vídeo de teste 1280×720 com tom de 1 kHz."""
    _run([ffmpeg, "-y", "-loglevel", "error",
          "-f", "lavfi", "-i", f"testsrc2=size=1280x720:rate=30:duration={seconds}",
          "-f", "lavfi", "-i", f"sine=frequency=1000:sample_rate=48000:duration={seconds}",
          "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
          "-c:a", "aac", "-shortest", str(out)])
    return out


def encode_ts(ffmpeg: str, src: Path, out: Path, mux_rate_bps: float, max_seconds: int = 20) -> float:
    """Codifica `src` em MPEG-TS cabendo em `mux_rate_bps`. Devolve a taxa de vídeo usada."""
    audio = 128_000 if mux_rate_bps > 1_000_000 else 48_000
    video = max(150_000, 0.82 * mux_rate_bps - audio)
    height = 720 if video > 2_500_000 else (480 if video > 800_000 else 240)
    _run([ffmpeg, "-y", "-loglevel", "error", "-i", str(src), "-t", str(max_seconds),
          "-map", "0:v:0", "-map", "0:a:0?",
          "-vf", f"scale=-2:'min({height},ih)'",
          "-c:v", "libx264", "-preset", "veryfast", "-profile:v", "high", "-pix_fmt", "yuv420p",
          "-b:v", str(int(video)), "-maxrate", str(int(video)), "-bufsize", str(int(video)),
          "-g", "30", "-c:a", "aac", "-b:a", str(audio), "-ac", "2",
          "-f", "mpegts", "-muxrate", str(int(mux_rate_bps)), str(out)])
    return video


def corrupt_ts(src: Path, dst: Path, per: float, byte_error: float, seed: int | None = None) -> TsStats:
    """Corrompe pacotes TS com probabilidade `per` (falha do RS)."""
    rng = np.random.default_rng(seed)
    data = np.frombuffer(src.read_bytes(), dtype=np.uint8).copy()
    n = data.size // P.TSP_BYTES
    pk = data[: n * P.TSP_BYTES].reshape(n, P.TSP_BYTES)
    if n and not np.all(pk[:, 0] == 0x47):
        raise ValueError("Arquivo não parece um TS de 188 bytes (sync 0x47 ausente).")
    pid = ((pk[:, 1].astype(np.int32) & 0x1F) << 8) | pk[:, 2]
    bad = rng.random(n) < per
    p_bad = max(byte_error, (P.RS_T + 1) / P.RS_N)
    idx = np.flatnonzero(bad)
    if idx.size:
        hits = rng.random((idx.size, P.TSP_BYTES - 4)) < p_bad
        noise = rng.integers(0, 256, hits.shape, dtype=np.uint8)
        payload = pk[idx, 4:]
        pk[idx, 4:] = np.where(hits, noise, payload)
        pk[idx, 1] |= 0x80  # TEI
    dst.write_bytes(data.tobytes())
    by_pid = {}
    for p in np.unique(pid):
        sel = pid == p
        by_pid[int(p)] = (int(sel.sum()), int(bad[sel].sum()))
    return TsStats(packets=int(n), corrupted=int(bad.sum()), per_target=per, by_pid=by_pid)


def decode_for_view(ffmpeg: str, ts: Path, out: Path, fps: float) -> None:
    """Decodifica o TS (tolerando erros) para MJPEG em CFR, para exibição."""
    _run([ffmpeg, "-y", "-loglevel", "error", "-err_detect", "ignore_err", "-fflags", "+genpts",
          "-i", str(ts), "-map", "0:v:0", "-an", "-fps_mode", "cfr", "-r", f"{fps:.3f}",
          "-c:v", "mjpeg", "-q:v", "4", str(out)])


def probe_fps(path: Path) -> float:
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        cap.release()
        if 1.0 < fps < 121.0:
            return float(fps)
    except Exception:  # noqa: BLE001
        pass
    return 30.0


def simulate_video(src: Path | None, mux_rate_bps: float, per: float, byte_error: float,
                   workdir: Path | None = None, progress: Progress | None = None,
                   seed: int | None = None, max_seconds: int = 20) -> VideoSimResult:
    """Executa a simulação completa. `src=None` usa um vídeo de teste gerado."""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("FFmpeg não encontrado. Instale com: sudo apt install ffmpeg")
    work = Path(workdir or tempfile.mkdtemp(prefix="isdbt_video_"))
    work.mkdir(parents=True, exist_ok=True)

    def step(pct: int, msg: str) -> None:
        if progress:
            progress(pct, msg)

    if src is None:
        step(5, "Gerando vídeo de teste…")
        src = make_test_video(ffmpeg, work / "teste.mp4")
    step(15, "Codificando em H.264 / MPEG-TS…")
    orig = work / "original.ts"
    vrate = encode_ts(ffmpeg, Path(src), orig, mux_rate_bps, max_seconds)
    step(50, "Corrompendo pacotes conforme a PER…")
    bad = work / "corrompido.ts"
    stats = corrupt_ts(orig, bad, per, byte_error, seed)
    fps = probe_fps(orig)
    step(60, "Decodificando original…")
    ov = work / "original_view.avi"
    decode_for_view(ffmpeg, orig, ov, fps)
    step(80, "Decodificando recebido…")
    cv = work / "corrompido_view.avi"
    decode_for_view(ffmpeg, bad, cv, fps)
    step(100, "Pronto")
    return VideoSimResult(orig, bad, ov, cv, stats, fps, vrate)
