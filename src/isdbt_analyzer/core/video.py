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

import os
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
    psi_pids: tuple = ()

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
    no_signal: bool = False  # nada decodificável: o receptor mostra "SEM SINAL"


FFMPEG_TIMEOUT_S = 180
NO_SIGNAL_PER = 0.9  # acima disso o receptor perde o sincronismo


def _lower_priority() -> None:  # roda no processo filho, antes do FFmpeg
    try:
        os.nice(10)
    except OSError:
        pass


_active: set[subprocess.Popen] = set()


def kill_active() -> None:
    """Encerra qualquer FFmpeg ainda rodando (chamado ao fechar o programa)."""
    for proc in list(_active):
        try:
            proc.kill()
        except OSError:
            pass
    _active.clear()


def _run(cmd: list[str], timeout: float = FFMPEG_TIMEOUT_S) -> None:
    """Roda o FFmpeg com prioridade baixa e tempo máximo (nunca deixa a simulação presa)."""
    if "-threads" not in cmd:
        cmd = cmd[:1] + ["-threads", "2"] + cmd[1:]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                            preexec_fn=_lower_priority if os.name == "posix" else None)
    _active.add(proc)
    try:
        _, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        proc.kill()
        proc.communicate()
        raise RuntimeError(f"O FFmpeg passou de {timeout:.0f} s e foi interrompido.") from exc
    finally:
        _active.discard(proc)
    if proc.returncode != 0:
        tail = "\n".join((stderr or "").strip().splitlines()[-8:])
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


def psi_pids(pk: np.ndarray, pid: np.ndarray) -> set[int]:
    """PIDs da PAT (0) e das PMTs, lidos da primeira PAT do arquivo."""
    out = {0}
    for row in pk[pid == 0]:
        if not row[1] & 0x40:          # sem payload_unit_start
            continue
        afc = (row[3] >> 4) & 0x3
        pos = 4
        if afc & 0x2:
            pos += 1 + int(row[4])
        pos += 1 + int(row[pos])       # pointer_field
        sec_len = ((int(row[pos + 1]) & 0x0F) << 8) | int(row[pos + 2])
        start, end = pos + 8, pos + 3 + sec_len - 4  # laço de programas (sem CRC)
        for i in range(start, min(end, P.TSP_BYTES - 3), 4):
            prog = (int(row[i]) << 8) | int(row[i + 1])
            pmt = ((int(row[i + 2]) & 0x1F) << 8) | int(row[i + 3])
            if prog != 0:
                out.add(pmt)
        break
    return out


def corrupt_ts(src: Path, dst: Path, per: float, byte_error: float, seed: int | None = None,
               keep_psi: bool = True) -> TsStats:
    """Corrompe pacotes TS com probabilidade `per` (falha do RS).

    Com `keep_psi`, a PAT e as PMTs não são corrompidas: um receptor já
    sintonizado guarda essas tabelas em cache e continua sabendo que o vídeo é
    H.264, mesmo quando o sinal está péssimo.
    """
    rng = np.random.default_rng(seed)
    data = np.frombuffer(src.read_bytes(), dtype=np.uint8).copy()
    n = data.size // P.TSP_BYTES
    pk = data[: n * P.TSP_BYTES].reshape(n, P.TSP_BYTES)
    if n and not np.all(pk[:, 0] == 0x47):
        raise ValueError("Arquivo não parece um TS de 188 bytes (sync 0x47 ausente).")
    pid = ((pk[:, 1].astype(np.int32) & 0x1F) << 8) | pk[:, 2]
    bad = rng.random(n) < per
    keep = psi_pids(pk, pid) if keep_psi else set()
    if keep:
        bad &= ~np.isin(pid, list(keep))
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
    return TsStats(packets=int(n), corrupted=int(bad.sum()), per_target=per, by_pid=by_pid,
                   psi_pids=tuple(sorted(keep)))


def decode_for_view(ffmpeg: str, ts: Path, out: Path, fps: float, max_frames: int | None = None) -> None:
    """Decodifica o TS (tolerando erros) para MJPEG em CFR, para exibição.

    Um pacote corrompido pode estragar o PTS e fazer o modo CFR "preencher" um
    buraco enorme com quadros repetidos. Por isso, a saída é limitada ao número
    de quadros do original.
    """
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-err_detect", "ignore_err", "-fflags", "+genpts",
           "-c:v", "h264",                 # o receptor sabe que é H.264 (não confia na sondagem)
           "-i", str(ts), "-map", "0:v:0", "-an", "-fps_mode", "cfr", "-r", f"{fps:.3f}",
           "-max_error_rate", "1.0"]       # não desiste por excesso de erros de decodificação
    if max_frames:
        cmd += ["-frames:v", str(int(max_frames))]
    cmd += ["-c:v", "mjpeg", "-q:v", "4", str(out)]
    _run(cmd)


def write_no_signal(out: Path, n_frames: int, fps: float, size: tuple[int, int] = (1280, 720)) -> None:
    """Grava um vídeo preto com "SEM SINAL", como a tela da TV quando o sinal some."""
    import cv2
    w, h = size
    frame = np.zeros((h, w, 3), np.uint8)
    text = "SEM SINAL"
    scale = max(1.0, w / 640)
    thick = max(2, int(scale * 2))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale * 1.6, thick)
    cv2.putText(frame, text, ((w - tw) // 2, (h + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, scale * 1.6,
                (230, 230, 230), thick, cv2.LINE_AA)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"MJPG"), float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError("Não foi possível gravar o vídeo de 'sem sinal'.")
    for _ in range(max(1, n_frames)):
        writer.write(frame)
    writer.release()


def frame_size(path: Path) -> tuple[int, int]:
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if w > 0 and h > 0:
            return w, h
    except Exception:  # noqa: BLE001
        pass
    return 1280, 720


def count_frames(path: Path) -> int:
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        return n
    except Exception:  # noqa: BLE001
        return 0


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
    n_orig = count_frames(ov) or int(fps * max_seconds)
    # com perda maciça o receptor perde o sincronismo e mostra "sem sinal"
    no_signal = stats.per >= NO_SIGNAL_PER
    if not no_signal:
        try:
            decode_for_view(ffmpeg, bad, cv, fps, max_frames=n_orig)
            no_signal = count_frames(cv) == 0
        except RuntimeError:
            no_signal = True  # nada decodificável
    if no_signal:
        cv.unlink(missing_ok=True)
        write_no_signal(cv, n_orig, fps, frame_size(ov))
    step(100, "Pronto")
    return VideoSimResult(orig, bad, ov, cv, stats, fps, vrate, no_signal)
