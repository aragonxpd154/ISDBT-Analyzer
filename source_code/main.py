"""
Graphical User Interface for ISDB-T Performance Prediction
=========================================================

GUI (Tkinter) para prever desempenho ISDB-T usando isdbt_tool.
Mantém layout original, preserva COR em imagem e vídeo (degrada
somente luminância Y em YCbCr), aplica curto-circuito em Eb/N0 alto
e estabiliza o vídeo com SNR travado + suavização temporal (EMA).

Uso:
    python isdbt_gui_obel_1.2.0.py
"""

__APP_NAME__ = "Previsor de Desempenho ISDB-T"
__AUTHOR__   = "Marcos Silva dos Santos"
__VERSION__  = "1.8.0"

import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.filedialog import askopenfilename
from tkinter import filedialog

from PIL import Image, ImageTk, ImageOps
import numpy as np

# Vídeo
import cv2

# Gráficos
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Módulo de cálculo
import isdbt_tool


class ISDBTGui(tk.Tk):
    """Janela principal da interface gráfica do ISDB-T."""

    # ===== Menubar / Sobre =====
    def _build_menubar(self):
        menubar = tk.Menu(self)
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Sobre…", command=self._show_about)
        menubar.add_cascade(label="Ajuda", menu=help_menu)
        self.config(menu=menubar)

    def _show_about(self):
        about_txt = (
            f"{__APP_NAME__}\n"
            f"Versão: {__VERSION__}\n"
            f"Desenvolvido por: {__AUTHOR__}"
        )
        messagebox.showinfo("Sobre", about_txt)

    # ===== Status bar =====
    def _build_statusbar(self):
        cols, rows = self.grid_size()
        if cols == 0:
            cols = 1
        sep = ttk.Separator(self, orient="horizontal")
        sep.grid(row=rows, column=0, columnspan=cols, sticky="ew")

        self.status_frame = ttk.Frame(self)
        self.status_frame.grid(row=rows + 1, column=0, columnspan=cols, sticky="ew")

        self.grid_columnconfigure(0, weight=1)
        self.status_frame.columnconfigure(0, weight=1)

        self.status_var = tk.StringVar(
            value=f"{__APP_NAME__}  •  v{__VERSION__}  •  {__AUTHOR__}"
        )
        status = ttk.Label(self.status_frame, textvariable=self.status_var, anchor="w")
        status.pack(side="left", padx=6, pady=2)

    # ===== Compat: nomes de modulação do isdbt_tool =====
    def _mod_aliases(self, base: str):
        """Gera aliases comuns para o nome de modulação esperado pelo isdbt_tool."""
        t = (base or "").strip().upper().replace(" ", "")
        t = t.replace("–", "-").replace("—", "-")  # hífens tipográficos -> ASCII
        if t in ("QPSK",):
            return ["QPSK", "qpsk"]
        if t in ("16QAM", "16-QAM"):
            return ["16-QAM", "16QAM", "16 QAM", "QAM16", "qam16"]
        if t in ("64QAM", "64-QAM"):
            return ["64-QAM", "64QAM", "64 QAM", "QAM64", "qam64"]
        return [base, t, t.replace("-", ""), t.replace("-", " "), t.replace(" ", "")]

    def _call_with_mod_aliases(self, func, modulation_base: str, *args, **kwargs):
        """Tenta func(mod, *args) com aliases até uma funcionar."""
        last_err = None
        for mod in self._mod_aliases(modulation_base):
            try:
                return func(mod, *args, **kwargs)
            except Exception as e:
                if isinstance(e, ValueError) and "Unsupported modulation" in str(e):
                    last_err = e
                    continue
                raise
        if last_err:
            raise last_err
        raise ValueError(f"Unsupported modulation (aliases tried) for '{modulation_base}'")

    # Wrappers seguros para o isdbt_tool (usados em todo lugar)
    def _theoretical_ber_safe(self, modulation: str, ebn0_range):
        return self._call_with_mod_aliases(isdbt_tool.theoretical_ber, modulation, ebn0_range)

    def _compute_data_rate_safe(self, modulation: str, coding_rate, guard: str):
        return self._call_with_mod_aliases(
            isdbt_tool.compute_data_rate, modulation, coding_rate, isdbt_tool.GUARD_INTERVALS[guard]
        )

    def _degrade_image_safe(self, img_pil_L, modulation: str, snr: float):
        """
        Chama isdbt_tool.degrade_image(img, modulation, snr) tentando aliases.
        (ORDEM CERTA: imagem primeiro, depois modulação, depois SNR)
        """
        last_err = None
        for alias in self._mod_aliases(modulation):
            try:
                return isdbt_tool.degrade_image(img_pil_L, alias, snr)
            except Exception as e:
                if isinstance(e, ValueError) and "Unsupported modulation" in str(e):
                    last_err = e
                    continue
                raise
        if last_err:
            raise last_err
        raise ValueError(f"Unsupported modulation (aliases tried) for '{modulation}'")

    # ===== Util =====
    def _norm_modulation(self) -> str:
        """
        Normaliza texto da combobox para algo consistente (QPSK, 16-QAM, 64-QAM).
        """
        t = (self.modulation_var.get() or "").strip().upper()
        t = t.replace("–", "-").replace("—", "-")
        if t in ("QPSK",):
            return "QPSK"
        if t in ("16-QAM", "16QAM"):
            return "16-QAM"
        if t in ("64-QAM", "64QAM"):
            return "64-QAM"
        return t

    def __init__(self):
        super().__init__()
        # Título da janela principal
        self.title(f"{__APP_NAME__} — v{__VERSION__} (por {__AUTHOR__})")
        self.resizable(False, False)

        # Limites máximos só para EXIBIÇÃO (não afetam imagem/vídeo original)
        self.MAX_DISPLAY_W = 900
        self.MAX_DISPLAY_H = 500

        # Estado de arquivos
        self.image_path = None
        self.video_path = None

        # Estado de vídeo
        self._cap = None
        self._video_window = None
        self._video_label = None
        self._video_photo = None
        self._video_paused = False
        self._video_target_fps = None

        # Estabilização de vídeo
        self._video_latched_snr = None   # SNR travado ao abrir
        self._y_prev = None              # EMA da luminância
        self._ema_alpha = 0.25           # fator da média exponencial (0.2–0.4 bom)

        self.create_widgets()
        self._build_statusbar()
        self._build_menubar()

    def create_widgets(self):
        """Create and arrange all widgets in the window (layout preservado)."""
        # Quadro para seleção de parâmetros
        param_frame = ttk.LabelFrame(self, text="Parâmetros de Transmissão")
        param_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        # Seleção de modo
        ttk.Label(param_frame, text="Modo:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.mode_var = tk.IntVar(value=3)
        mode_options = [1, 2, 3]
        self.mode_menu = ttk.OptionMenu(
            param_frame, self.mode_var, 3, *mode_options, command=self.update_interleaver_options
        )
        self.mode_menu.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # Seleção de intervalo de guarda
        ttk.Label(param_frame, text="Intervalo de guarda:").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.guard_var = tk.StringVar(value="1/8")
        guard_options = ["1/4", "1/8", "1/16", "1/32"]
        self.guard_menu = ttk.OptionMenu(param_frame, self.guard_var, "1/8", *guard_options)
        self.guard_menu.grid(row=0, column=3, padx=5, pady=5, sticky="w")

        # Seleção de modulação
        ttk.Label(param_frame, text="Modulação:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.modulation_var = tk.StringVar(value="16-QAM")
        modulation_options = ["QPSK", "16-QAM", "64-QAM"]
        self.mod_menu = ttk.OptionMenu(param_frame, self.modulation_var, "16-QAM", *modulation_options)
        self.mod_menu.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # Seleção de taxa de codificação
        ttk.Label(param_frame, text="Taxa de codificação:").grid(row=1, column=2, padx=5, pady=5, sticky="w")
        self.coding_var = tk.StringVar(value="3/4")
        coding_options = ["1/2", "2/3", "3/4", "5/6", "7/8"]
        self.coding_menu = ttk.OptionMenu(param_frame, self.coding_var, "3/4", *coding_options)
        self.coding_menu.grid(row=1, column=3, padx=5, pady=5, sticky="w")

        # Seleção de profundidade de interleaver (entrelaçamento)
        ttk.Label(param_frame, text="Profundidade de entrelaçamento:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.interleaver_var = tk.IntVar(value=isdbt_tool.MODES[3].interleaver_depths[0])
        self.interleaver_menu = ttk.OptionMenu(
            param_frame,
            self.interleaver_var,
            isdbt_tool.MODES[3].interleaver_depths[0],
            *isdbt_tool.MODES[3].interleaver_depths
        )
        self.interleaver_menu.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        # Faixa de Eb/N0
        ttk.Label(param_frame, text="Início de Eb/N0 (dB):").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.eb_start = tk.DoubleVar(value=0.0)
        ttk.Entry(param_frame, textvariable=self.eb_start, width=8).grid(row=3, column=1, padx=5, pady=5)

        ttk.Label(param_frame, text="Fim de Eb/N0 (dB):").grid(row=3, column=2, padx=5, pady=5, sticky="w")
        self.eb_end = tk.DoubleVar(value=15.0)
        ttk.Entry(param_frame, textvariable=self.eb_end, width=8).grid(row=3, column=3, padx=5, pady=5)

        ttk.Label(param_frame, text="Passo de Eb/N0 (dB):").grid(row=3, column=4, padx=5, pady=5, sticky="w")
        self.eb_step = tk.DoubleVar(value=5.0)
        ttk.Entry(param_frame, textvariable=self.eb_step, width=6).grid(row=3, column=5, padx=5, pady=5)

        # Seleção de IMAGEM
        self.image_path = None
        ttk.Button(param_frame, text="Selecionar imagem", command=self.select_image).grid(
            row=4, column=0, columnspan=2, padx=5, pady=5
        )
        self.image_label = ttk.Label(
            param_frame,
            text="Nenhuma imagem selecionada (o gradiente padrão será utilizado)"
        )
        self.image_label.grid(row=4, column=2, columnspan=4, padx=5, pady=5, sticky="w")

        # Seleção de VÍDEO
        self.video_path = None
        ttk.Button(param_frame, text="Selecionar vídeo", command=self.open_video).grid(
            row=5, column=0, columnspan=2, padx=5, pady=5
        )
        self.video_label = ttk.Label(param_frame, text="Nenhum vídeo selecionado")
        self.video_label.grid(row=5, column=2, columnspan=4, padx=5, pady=5, sticky="w")

        # Botão para executar a simulação (layout preservado)
        ttk.Button(self, text="Executar simulação", command=self.run_simulation).grid(
            row=1, column=0, padx=10, pady=(0, 10)
        )

        # Separador (layout preservado)
        sep = ttk.Separator(self, orient="horizontal")
        sep.grid(row=11, column=0, columnspan=2, sticky="ew", pady=(10, 10))

        # Botão Sobre (layout preservado)
        ttk.Button(self, text="Sobre", command=self.show_about).grid(row=2, column=0, padx=10, pady=(0, 10))

    def update_interleaver_options(self, _):
        """Atualiza opções do interleaver conforme o modo."""
        mode = self.mode_var.get()
        depths = isdbt_tool.MODES[mode].interleaver_depths
        menu = self.interleaver_menu['menu']
        menu.delete(0, 'end')
        for depth in depths:
            menu.add_command(label=str(depth), command=lambda d=depth: self.interleaver_var.set(d))
        self.interleaver_var.set(depths[0])

    def select_image(self):
        """Seleciona uma imagem (qualquer formato); exibiremos e degradaremos preservando cor."""
        path = askopenfilename(
            title="Selecionar imagem",
            filetypes=[("Arquivos de imagem", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")]
        )
        if path:
            try:
                Image.open(path)  # valida abertura
                self.image_path = path
                self.image_label.config(text=f"Imagem selecionada: {path}")
            except Exception as e:
                messagebox.showerror("Erro", f"Não foi possível abrir a imagem: {e}")
        else:
            self.image_path = None
            self.image_label.config(text="Nenhuma imagem selecionada (o gradiente padrão será utilizado)")

    # ============= Simulação com IMAGEM (colorida + curto-circuito p/ Eb/N0 alto) =============
    def run_simulation(self):
        """Executa cálculos e exibe: gráfico de BER + comparação da IMAGEM (colorida)."""
        try:
            guard = self.guard_var.get()
            modulation = self._norm_modulation()
            coding_rate_str = self.coding_var.get()
            coding_rate = eval(coding_rate_str)  # opções controladas
            eb_start = float(self.eb_start.get())
            eb_end   = float(self.eb_end.get())
            eb_step  = float(self.eb_step.get())
            if eb_step <= 0:
                raise ValueError("O passo de Eb/N0 deve ser positivo")

            ebn0_range = np.arange(eb_start, eb_end + 0.001, eb_step)

            # IMAGEM de referência: trabalharemos em RGB e degradaremos SOMENTE Y (YCbCr)
            if self.image_path:
                img_rgb = Image.open(self.image_path).convert('RGB')
            else:
                # Gradiente padrão como "RGB neutro": luminância em Y, cromas neutros (Cb/Cr 128)
                grad = np.linspace(0, 255, 256, dtype=np.uint8)
                y = Image.fromarray(np.tile(grad, (256, 1)), mode='L')  # 256x256 gradiente
                cb = Image.new('L', y.size, 128)
                cr = Image.new('L', y.size, 128)
                img_rgb = Image.merge("YCbCr", (y, cb, cr)).convert("RGB")

            # Curvas BER (compatíveis)
            ber_uncoded = self._theoretical_ber_safe(modulation, ebn0_range)
            gain_map = {0.5:3.0, 2/3:5.0, 3/4:6.0, 5/6:7.5, 7/8:8.5}
            gain_db = gain_map.get(coding_rate, 0.0)
            ber_coded = self._theoretical_ber_safe(modulation, ebn0_range + gain_db)

            # Janela de BER
            plot_window = tk.Toplevel(self)
            plot_window.title("Curva de BER")
            fig = Figure(figsize=(6, 4), dpi=100)
            ax = fig.add_subplot(111)
            ax.semilogy(ebn0_range, ber_uncoded, 'o-', label='BER sem codificação')
            ax.semilogy(ebn0_range, ber_coded, 's--', label='BER codificada (aprox.)')
            ax.set_title(f'BER x Eb/N0 ({modulation}, R={coding_rate}, GI={guard})')
            ax.set_xlabel('Eb/N0 (dB)')
            ax.set_ylabel('Taxa de erro de bit')
            ax.grid(True, which='both')
            ax.legend()
            canvas = FigureCanvasTkAgg(fig, master=plot_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)

            data_rate = self._compute_data_rate_safe(modulation, coding_rate, guard)
            ttk.Label(plot_window, text=f'Taxa de dados aproximada: {data_rate:.2f} Mb/s').pack(pady=5)

            # Janela de comparação de IMAGENS (usa primeiro e último Eb/N0) – preservando COR
            img_window = tk.Toplevel(self)
            img_window.title("Comparação de imagens (Original × Degradada)")
            frame = ttk.Frame(img_window)
            frame.pack(padx=10, pady=10)

            self._photo_refs = []
            snr_list = [float(ebn0_range[0]), float(ebn0_range[-1])] if len(ebn0_range) > 1 else [float(ebn0_range[0])]
            for idx, snr in enumerate(snr_list):
                # Curto-circuito para Eb/N0 alto em IMAGEM
                ycbcr_src = img_rgb.convert("YCbCr")
                y_src, cb_src, cr_src = ycbcr_src.split()
                if snr >= 40.0:
                    y_deg = y_src.copy()
                else:
                    y_deg = self._degrade_image_safe(y_src, modulation, snr)
                img_deg_rgb = Image.merge("YCbCr", (y_deg, cb_src, cr_src)).convert("RGB")

                w, h = img_rgb.size
                combined = Image.new('RGB', (w * 2 + 10, h))
                combined.paste(img_rgb, (0, 0))
                combined.paste(img_deg_rgb, (w + 10, 0))

                # Reduz somente para exibição (sem esticar/upscaling)
                display_img = ImageOps.contain(
                    combined,
                    (self.MAX_DISPLAY_W, self.MAX_DISPLAY_H),
                    method=Image.Resampling.LANCZOS
                )
                photo = ImageTk.PhotoImage(display_img)
                self._photo_refs.append(photo)
                tk.Label(frame, image=photo).grid(row=0, column=idx, padx=5, pady=5)
                ttk.Label(frame, text=f'Eb/N0 = {snr:.2f} dB').grid(row=1, column=idx, pady=(0, 10))

        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def show_about(self):
        """Exibe informações sobre os pacotes utilizados, licença e referência."""
        info = (
            "Esta aplicação utiliza as seguintes bibliotecas:\n"
            "- Tkinter para a interface gráfica\n"
            "- NumPy para cálculo numérico\n"
            "- Pillow para processamento de imagens\n"
            "- Matplotlib para gráficos\n"
            "- isdbt_tool (módulo auxiliar)\n\n"
            f"{__APP_NAME__} — v{__VERSION__}\n"
            f"Desenvolvido por: {__AUTHOR__}\n"
            "Licença: GPL v3\n"
            "Mais informações em: www.obellab.com.br"
        )
        messagebox.showinfo("Sobre", info)

    # ============= VÍDEO: seleção, gráfico de BER e exibição =============
    def open_video(self):
        """Seleciona um arquivo de vídeo e abre: janela de vídeo + gráfico de BER (vídeo)."""
        path = filedialog.askopenfilename(
            title="Selecione um arquivo de vídeo",
            filetypes=[("Vídeos", "*.mp4;*.avi;*.mkv;*.mov;*.webm"), ("Todos", "*.*")]
        )
        if not path:
            return

        self.video_path = path
        self.video_label.config(text=f"Vídeo selecionado: {path}")

        # Libera captura anterior
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            messagebox.showerror("Erro", "Não foi possível abrir o vídeo.")
            return

        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._video_target_fps = fps if fps and fps > 0 else 30.0

        # --- TRAVA Eb/N0 do vídeo no momento da abertura ---
        try:
            self._video_latched_snr = float(self.eb_start.get())  # troque para eb_end se preferir
        except Exception:
            self._video_latched_snr = 15.0

        self._open_video_window()
        self._video_paused = False
        self._y_prev = None  # limpa estado de EMA
        self._video_loop()

        # Abre também o gráfico de BER específico para a sessão de vídeo
        self._open_video_ber_window()

    def _open_video_window(self):
        if self._video_window and self._video_window.winfo_exists():
            self._video_window.destroy()

        self._video_window = tk.Toplevel(self)
        self._video_window.title("Comparação de vídeo (Original × Degradado)")
        self._video_window.resizable(False, False)
        self._video_window.protocol("WM_DELETE_WINDOW", self._close_video_window)

        self._video_label = ttk.Label(self._video_window)
        self._video_label.pack(padx=10, pady=10)

        ctrl = ttk.Frame(self._video_window)
        ctrl.pack(fill="x", padx=10, pady=(0, 10))

        ttk.Button(ctrl, text="Pausar/Retomar", command=self._toggle_pause).pack(side="left")
        ttk.Button(ctrl, text="Parar", command=self._close_video_window).pack(side="left", padx=8)

    def _open_video_ber_window(self):
        """Abre um gráfico de BER baseado nos parâmetros atuais (para a sessão de vídeo)."""
        try:
            guard = self.guard_var.get()
            modulation = self._norm_modulation()
            coding_rate_str = self.coding_var.get()
            coding_rate = eval(coding_rate_str)

            eb_start = float(self.eb_start.get())
            eb_end   = float(self.eb_end.get())
            eb_step  = float(self.eb_step.get())
            if eb_step <= 0:
                raise ValueError("O passo de Eb/N0 deve ser positivo")

            ebn0_range = np.arange(eb_start, eb_end + 0.001, eb_step)

            ber_uncoded = self._theoretical_ber_safe(modulation, ebn0_range)
            gain_map = {0.5:3.0, 2/3:5.0, 3/4:6.0, 5/6:7.5, 7/8:8.5}
            gain_db = gain_map.get(coding_rate, 0.0)
            ber_coded = self._theoretical_ber_safe(modulation, ebn0_range + gain_db)

            plot_window = tk.Toplevel(self)
            plot_window.title("Curva de BER (Vídeo)")
            fig = Figure(figsize=(6, 4), dpi=100)
            ax = fig.add_subplot(111)
            ax.semilogy(ebn0_range, ber_uncoded, 'o-', label='BER sem codificação')
            ax.semilogy(ebn0_range, ber_coded, 's--', label='BER codificada (aprox.)')
            ax.set_title(f'BER x Eb/N0 (Vídeo) — {modulation}, R={coding_rate}, GI={guard}')
            ax.set_xlabel('Eb/N0 (dB)')
            ax.set_ylabel('Taxa de erro de bit')
            ax.grid(True, which='both')
            ax.legend()
            canvas = FigureCanvasTkAgg(fig, master=plot_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)

            data_rate = self._compute_data_rate_safe(modulation, coding_rate, guard)
            ttk.Label(plot_window, text=f'Taxa de dados aproximada: {data_rate:.2f} Mb/s').pack(pady=5)

        except Exception as e:
            messagebox.showerror("Erro (vídeo/BER)", str(e))

    def _toggle_pause(self):
        self._video_paused = not self._video_paused

    def _close_video_window(self):
        self._video_paused = True
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        if self._video_window and self._video_window.winfo_exists():
            self._video_window.destroy()
        self._video_window = None
        self._video_label = None
        self._video_photo = None
        # limpa estados de vídeo
        self._y_prev = None
        self._video_latched_snr = None

    def _video_loop(self):
        """Loop de exibição do vídeo: degrada SOMENTE Y (luminância), SNR travado + EMA."""
        if self._video_window is None or self._cap is None:
            return
        if not self._video_window.winfo_exists():
            self._close_video_window()
            return

        delay_ms = int(1000.0 / (self._video_target_fps or 30.0))

        if self._video_paused:
            self.after(delay_ms, self._video_loop)
            return

        ok, frame = self._cap.read()
        if not ok:
            # Loop do vídeo
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.after(delay_ms, self._video_loop)
            return

        # OpenCV BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_orig_rgb = Image.fromarray(frame_rgb)  # RGB

        # Separa canais Y, Cb, Cr
        ycbcr = pil_orig_rgb.convert("YCbCr")
        y, cb, cr = ycbcr.split()  # y é 'L' (grayscale)

        # Usa SNR TRAVADO (não muda durante a reprodução)
        try:
            snr = float(self._video_latched_snr if self._video_latched_snr is not None else 15.0)
        except Exception:
            snr = 15.0

        modulation = self._norm_modulation()

        # Curto-circuito para Eb/N0 muito alto (vídeo idêntico ao original)
        if snr >= 40.0:
            y_deg = y.copy()
        else:
            y_deg = self._degrade_image_safe(y, modulation, snr)

        # EMA temporal na luminância (suaviza cintilação)
        if self._y_prev is None:
            y_smooth = y_deg
        else:
            y_arr = np.asarray(y_deg, dtype=np.float32)
            prev_arr = np.asarray(self._y_prev, dtype=np.float32)
            mix = (self._ema_alpha * y_arr + (1.0 - self._ema_alpha) * prev_arr).clip(0, 255).astype(np.uint8)
            y_smooth = Image.fromarray(mix, mode="L")
        self._y_prev = y_smooth

        # Recompõe o degradado em cor
        ycbcr_deg = Image.merge("YCbCr", (y_smooth, cb, cr))
        pil_deg_rgb = ycbcr_deg.convert("RGB")

        # Monta comparação lado a lado (RGB)
        w, h = pil_orig_rgb.size
        combined = Image.new('RGB', (w * 2 + 10, h))
        combined.paste(pil_orig_rgb, (0, 0))
        combined.paste(pil_deg_rgb, (w + 10, 0))

        # Reduz só para EXIBIÇÃO (sem esticar, sem upscaling)
        display_img = ImageOps.contain(
            combined,
            (self.MAX_DISPLAY_W, self.MAX_DISPLAY_H),
            method=Image.Resampling.LANCZOS
        )

        self._video_photo = ImageTk.PhotoImage(display_img)
        if self._video_label and self._video_label.winfo_exists():
            self._video_label.configure(image=self._video_photo)

        self.after(delay_ms, self._video_loop)


if __name__ == '__main__':
    app = ISDBTGui()
    app.mainloop()
