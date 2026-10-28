"""Ponto de entrada: sem argumentos abre a interface; com subcomandos funciona no terminal.

Exemplos:
    isdbt-analyzer                                   # abre a interface gráfica
    isdbt-analyzer taxa --modo 3 --guarda 1/16 -c A:1:QPSK:2/3:4 -c B:12:64QAM:3/4:2
    isdbt-analyzer desempenho -c B:12:64QAM:3/4 --canal "Brasil A" --cn 20
    isdbt-analyzer curvas -c B:12:64QAM:3/4 -o curvas.csv
    isdbt-analyzer relatorio -o relatorio.pdf --cn 22
    isdbt-analyzer imagem foto.jpg --cn 17 -c B:12:64QAM:3/4 -o recebida.png
    isdbt-analyzer video entrada.mp4 --cn 18.4 -c B:12:64QAM:3/4 -o saida/
    isdbt-analyzer calibrar                          # refaz as tabelas de FEC (demorado)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __app_name__, __version__


def _cfg_from_args(args):
    from .core import config_io
    from .core.ofdm import TransmissionConfig
    if getattr(args, "config", None):
        cfg, ch, cn = config_io.load(args.config)
        return cfg, (args.canal or ch), (args.cn if args.cn is not None else cn)
    layers = tuple(config_io.parse_layer(s) for s in (args.camada or [])) or None
    kw = {"mode": args.modo, "guard": args.guarda}
    if layers:
        kw["layers"] = layers
    cfg = TransmissionConfig(**kw)
    for w in cfg.validate():
        print(f"aviso: {w}", file=sys.stderr)
    return cfg, (args.canal or "AWGN"), args.cn


def _add_cfg_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--modo", type=int, default=3, choices=(1, 2, 3))
    p.add_argument("--guarda", default="1/16", choices=("1/4", "1/8", "1/16", "1/32"))
    p.add_argument("-c", "--camada", action="append",
                   help="NOME:SEG:MOD:CODIGO[:I], ex.: B:12:64QAM:3/4:2 (repita para várias camadas)")
    p.add_argument("--canal", default=None, help="AWGN, 'Brasil A' … 'Brasil E'")
    p.add_argument("--cn", type=float, default=None, help="C/N de operação (dB)")
    p.add_argument("--config", help="arquivo .json salvo pela interface")


def cmd_taxa(args) -> int:
    cfg, _, _ = _cfg_from_args(args)
    s = cfg.summary()
    print(f"Modo {cfg.mode} · GI {cfg.guard} · Tu {s['Tu_us']:.1f} µs · Tg {s['Tg_us']:.1f} µs · "
          f"Δf {s['espacamento_Hz']:.2f} Hz · SFN máx {s['sfn_km']:.1f} km")
    print(f"{'Camada':<7}{'Seg':>4}  {'Mod':<7}{'Cód':<5}{'I':>2}  {'Mb/s':>8}  {'TSP/quadro':>10}  {'atraso TI':>9}")
    for c in s["camadas"]:
        print(f"{c['camada']:<7}{c['segmentos']:>4}  {c['modulacao']:<7}{c['codigo']:<5}{c['I']:>2}  "
              f"{c['taxa_Mbps']:>8.3f}  {c['tsp_por_quadro']:>10.0f}  {c['atraso_entrelacador_ms']:>7.0f} ms")
    print(f"{'Total':<29}{s['taxa_total_Mbps']:>8.3f} Mb/s")
    return 0


def cmd_desempenho(args) -> int:
    from .core.analysis import fmt_prob, layer_performance
    cfg, ch, cn = _cfg_from_args(args)
    print(f"Canal: {ch}" + (f" · C/N = {cn:.1f} dB" if cn is not None else ""))
    for p in layer_performance(cfg, ch, cn):
        line = (f"Camada {p.layer.name}: {p.layer.modulation} {p.layer.code_rate} → "
                f"{p.rate_bps / 1e6:.3f} Mb/s · C/N QEF {p.qef_esn0_db:.1f} dB "
                f"(Eb/N0 {p.qef_ebn0_db:.1f} dB)")
        if cn is not None:
            m = p.margin(cn)
            status = "OK" if m >= 0 else "FALHA"
            line += (f" · margem {m:+.1f} dB [{status}] · BER bruta {fmt_prob(p.raw_ber_at, 2)}"
                     f" · BER Viterbi {fmt_prob(p.vit_ber_at, 2)} · PER {fmt_prob(p.per_at, 2)}")
        print(line)
    return 0


def cmd_curvas(args) -> int:
    from .core.analysis import export_csv
    cfg, ch, _ = _cfg_from_args(args)
    out = export_csv(args.output, cfg, ch, args.de, args.ate, args.passo)
    print(f"CSV salvo em {out}")
    return 0


def cmd_relatorio(args) -> int:
    from .core.report import export_pdf
    cfg, ch, cn = _cfg_from_args(args)
    out = export_pdf(args.output, cfg, ch, cn)
    print(f"Relatório salvo em {out}")
    return 0


def _layer_and_curve(cfg, ch, which):
    from .core.tables import get_curve
    lay = next((lay for lay in cfg.layers if lay.name == which), cfg.layers[-1]).normalized()
    return lay, get_curve(lay.modulation, lay.code_rate, ch)


def cmd_imagem(args) -> int:
    from PIL import Image

    from .core import media
    from .core.channel import PROFILES, ChannelState
    cfg, ch, cn = _cfg_from_args(args)
    if cn is None:
        print("Informe --cn", file=sys.stderr)
        return 2
    img = media.prepare(Image.open(args.arquivo)) if args.arquivo else media.test_pattern()
    lay, curve = _layer_and_curve(cfg, ch, args.camada_alvo)
    if args.sem_fec:
        out, st = media.degrade_uncoded(img, lay.modulation, cn, ChannelState.build(PROFILES[ch], cfg),
                                        luma_only=not args.rgb)
    else:
        out, st = media.degrade_coded(img, float(curve.per(cn)[0]), float(curve.byte_error(cn)[0]),
                                      luma_only=not args.rgb)
    out.save(args.output)
    print(f"{st} → {args.output}")
    return 0


def cmd_video(args) -> int:
    from .core import video
    cfg, ch, cn = _cfg_from_args(args)
    if cn is None:
        print("Informe --cn", file=sys.stderr)
        return 2
    lay, curve = _layer_and_curve(cfg, ch, args.camada_alvo)
    per, be = float(curve.per(cn)[0]), float(curve.byte_error(cn)[0])
    r = video.simulate_video(Path(args.arquivo) if args.arquivo else None, cfg.layer_rate(lay), per, be,
                             workdir=Path(args.output), progress=lambda p, m: print(f"{p:3d}% {m}"))
    print(f"PER prevista {per:.2e} · pacotes {r.stats.packets} · corrompidos {r.stats.corrupted}")
    print(f"TS original: {r.original_ts}\nTS corrompido: {r.corrupted_ts}")
    return 0


def cmd_calibrar(args) -> int:
    from .core import tables
    out = Path(args.output) if args.output else Path(__file__).parent / "resources" / tables.TABLE_FILE
    tables.calibrate(out, args.canais or None, args.pacotes, args.erros)
    print(f"Tabelas gravadas em {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="isdbt-analyzer", description=f"{__app_name__} {__version__}",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("taxa", help="taxas por camada e parâmetros OFDM")
    _add_cfg_args(s)
    s.set_defaults(func=cmd_taxa)

    s = sub.add_parser("desempenho", help="C/N mínimo (QEF), margem e PER por camada")
    _add_cfg_args(s)
    s.set_defaults(func=cmd_desempenho)

    s = sub.add_parser("curvas", help="exporta curvas BER/PER em CSV")
    _add_cfg_args(s)
    s.add_argument("-o", "--output", default="curvas.csv")
    s.add_argument("--de", type=float, default=0.0)
    s.add_argument("--ate", type=float, default=30.0)
    s.add_argument("--passo", type=float, default=0.25)
    s.set_defaults(func=cmd_curvas)

    s = sub.add_parser("relatorio", help="gera relatório PDF")
    _add_cfg_args(s)
    s.add_argument("-o", "--output", default="relatorio_isdbt.pdf")
    s.set_defaults(func=cmd_relatorio)

    s = sub.add_parser("imagem", help="degrada uma imagem")
    _add_cfg_args(s)
    s.add_argument("arquivo", nargs="?", help="imagem (omita para usar o padrão de teste)")
    s.add_argument("-o", "--output", default="recebida.png")
    s.add_argument("--camada-alvo", default="B")
    s.add_argument("--sem-fec", action="store_true", help="sem Viterbi/RS (erros de bit crus)")
    s.add_argument("--rgb", action="store_true", help="corrompe RGB em vez de só a luminância")
    s.set_defaults(func=cmd_imagem)

    s = sub.add_parser("video", help="simula vídeo em Transport Stream")
    _add_cfg_args(s)
    s.add_argument("arquivo", nargs="?", help="vídeo (omita para gerar um de teste)")
    s.add_argument("-o", "--output", default="simulacao_video")
    s.add_argument("--camada-alvo", default="B")
    s.set_defaults(func=cmd_video)

    s = sub.add_parser("calibrar", help="refaz as tabelas de FEC por Monte Carlo (demorado)")
    s.add_argument("-o", "--output")
    s.add_argument("--canais", nargs="*")
    s.add_argument("--pacotes", type=int, default=2500)
    s.add_argument("--erros", type=int, default=300)
    s.set_defaults(func=cmd_calibrar)

    s = sub.add_parser("gui", help="abre a interface gráfica (padrão)")
    s.set_defaults(func=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not getattr(args, "func", None):
        from .gui.app import run
        return run()
    try:
        return args.func(args)
    except (ValueError, KeyError, RuntimeError, FileNotFoundError) as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

