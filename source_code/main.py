"""
Graphical User Interface for ISDB‑T Performance Prediction
=========================================================

This module provides a desktop GUI application built with Tkinter that
presents a friendly interface to the simplified ISDB‑T performance
prediction tool.  It allows users to select transmission parameters,
compute theoretical BER curves, estimate data rates, and visualise the
impact of noise on a reference image.  When the user clicks the
"Run Simulation" button, the program opens new windows displaying
a BER plot and an image comparison (original and degraded) for the
selected Eb/N0 range.

The GUI uses the functions defined in :mod:`isdbt_tool` for the
underlying calculations.  Make sure that `isdbt_tool.py` is in the same
directory or on the Python path when running this script.

This application is intended to be run on a local machine with a
desktop environment.  When executed in an environment without GUI
support (e.g., a headless server), it will raise an exception.

Usage::

    python isdbt_gui.py

"""

__APP_NAME__ = "Previsor de Desempenho ISDB-T"
__AUTHOR__   = "Marcos Silva dos Santos"
__VERSION__  = "1.1.0"

import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk, ImageOps

import numpy as np
from PIL import Image, ImageTk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import isdbt_tool


class ISDBTGui(tk.Tk):
    """Janela principal da interface gráfica do ISDB‑T.

    Todos os textos, botões e mensagens são apresentados em português
    para facilitar o uso pelos profissionais de radiodifusão. A GUI
    permite selecionar parâmetros, executar a simulação e abrir
    janelas para visualizar os resultados. Um botão "Sobre" fornece
    informações sobre as bibliotecas utilizadas, a licença GPL v3 e
    um link de referência.
    """

    def _build_menubar(self):
        menubar = tk.Menu(self)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Sobre…", command=self._show_about)
        menubar.add_cascade(label="Ajuda", menu=help_menu)

    def _show_about(self):
        about_txt = (
            f"{__APP_NAME__}\n"
            f"Versão: {__VERSION__}\n"
            f"Desenvolvido por: {__AUTHOR__}"
        )
        messagebox.showinfo("Sobre", about_txt)
        self.config(menu=menubar)
    
    def _build_statusbar(self):
        # Descobre quantas colunas/linhas já existem na raiz
        cols, rows = self.grid_size()
        if cols == 0:
            cols = 1  # fallback

        # Frame da barra de status, posicionado com GRID na última linha
        self.status_frame = ttk.Frame(self)
        self.status_frame.grid(row=rows, column=0, columnspan=cols, sticky="ew")
        # Faz a linha/coluna esticarem
        self.grid_columnconfigure(0, weight=1)
        self.status_frame.columnconfigure(0, weight=1)

        # (Opcional) separador visual acima da barra
        sep = ttk.Separator(self, orient="horizontal")
        sep.grid(row=rows-1, column=0, columnspan=cols, sticky="ew")

        # Conteúdo da barra de status — aqui dentro, pode usar PACK à vontade
        self.status_var = tk.StringVar(
            value=f"{__APP_NAME__}  •  v{__VERSION__}  •  {__AUTHOR__}"
        )
        status = ttk.Label(self.status_frame, textvariable=self.status_var, anchor="w")
        status.pack(side="left", padx=6, pady=2)

    def __init__(self):
        super().__init__()
        # Título da janela principal
        self.title(f"{__APP_NAME__} — v{__VERSION__} (por {__AUTHOR__})")
        self.resizable(False, False)
        # >>> Limites máximos para a EXIBIÇÃO (não afetam a imagem original)
        self.MAX_DISPLAY_W = 900   # ajuste maximo
        self.MAX_DISPLAY_H = 500   # ajuste maximo
        self.create_widgets()
        # barra de status (rodapé)
        self._build_statusbar()
        self._build_menubar()

    def create_widgets(self):
        """Create and arrange all widgets in the window."""
        # Quadro para seleção de parâmetros
        param_frame = ttk.LabelFrame(self, text="Parâmetros de Transmissão")
        param_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        # Seleção de modo
        ttk.Label(param_frame, text="Modo:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.mode_var = tk.IntVar(value=3)
        mode_options = [1, 2, 3]
        self.mode_menu = ttk.OptionMenu(param_frame, self.mode_var, 3, *mode_options, command=self.update_interleaver_options)
        self.mode_menu.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # Seleção de intervalo de guarda
        ttk.Label(param_frame, text="Intervalo de guarda:").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.guard_var = tk.StringVar(value="1/8")
        guard_options = ["1/4", "1/8", "1/16", "1/32"]
        self.guard_menu = ttk.OptionMenu(param_frame, self.guard_var, "1/8", *guard_options)
        self.guard_menu.grid(row=0, column=3, padx=5, pady=5, sticky="w")

        # Seleção de modulação
        ttk.Label(param_frame, text="Modulação:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.modulation_var = tk.StringVar(value="16‑QAM")
        modulation_options = ["QPSK", "16‑QAM", "64‑QAM"]
        self.mod_menu = ttk.OptionMenu(param_frame, self.modulation_var, "16‑QAM", *modulation_options)
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
        self.interleaver_menu = ttk.OptionMenu(param_frame, self.interleaver_var, isdbt_tool.MODES[3].interleaver_depths[0], *isdbt_tool.MODES[3].interleaver_depths)
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

        # Seleção de imagem
        self.image_path = None
        ttk.Button(param_frame, text="Selecionar imagem", command=self.select_image).grid(row=4, column=0, columnspan=2, padx=5, pady=5)
        self.image_label = ttk.Label(param_frame, text="Nenhuma imagem selecionada (o gradiente padrão será utilizado)")
        self.image_label.grid(row=4, column=2, columnspan=4, padx=5, pady=5, sticky="w")

        # Botão para executar a simulação
        ttk.Button(self, text="Executar simulação", command=self.run_simulation).grid(row=1, column=0, padx=10, pady=(0, 10))

        # Linha de separação (agora entre os botões)
        sep = ttk.Separator(self, orient="horizontal")
        sep.grid(row=11, column=0, columnspan=2, sticky="ew", pady=(10, 10))
        
        # Botão Sobre
        ttk.Button(self, text="Sobre", command=self.show_about).grid(row=2, column=0, padx=10, pady=(0, 10))

    def update_interleaver_options(self, _):
        """Update interleaver depth options based on selected mode."""
        mode = self.mode_var.get()
        depths = isdbt_tool.MODES[mode].interleaver_depths
        menu = self.interleaver_menu['menu']
        menu.delete(0, 'end')
        for depth in depths:
            menu.add_command(label=str(depth), command=lambda d=depth: self.interleaver_var.set(d))
        self.interleaver_var.set(depths[0])

    def select_image(self):
        """Abre um diálogo para selecionar uma imagem em tons de cinza para transmissão."""
        # Título e filtro em português
        path = askopenfilename(title="Selecionar imagem", filetypes=[("Arquivos de imagem", "*.png;*.jpg;*.jpeg;*.bmp")])
        if path:
            try:
                # Verifica se a imagem pode ser carregada e convertida para tons de cinza
                img = Image.open(path).convert('L')
                self.image_path = path
                self.image_label.config(text=f"Imagem selecionada: {path}")
            except Exception as e:
                messagebox.showerror("Erro", f"Não foi possível abrir a imagem: {e}")
        else:
            # Nenhuma imagem escolhida; usar gradiente padrão
            self.image_path = None
            self.image_label.config(text="Nenhuma imagem selecionada (o gradiente padrão será utilizado)")

    def run_simulation(self):
        """Coleta parâmetros, executa os cálculos e exibe os resultados."""
        try:
            # Obter parâmetros selecionados na GUI
            mode = self.mode_var.get()
            guard = self.guard_var.get()
            modulation = self.modulation_var.get()
            coding_rate_str = self.coding_var.get()
            coding_rate = eval(coding_rate_str)
            interleaver = self.interleaver_var.get()
            eb_start = self.eb_start.get()
            eb_end = self.eb_end.get()
            eb_step = self.eb_step.get()
            if eb_step <= 0:
                raise ValueError("O passo de Eb/N0 deve ser positivo")

            ebn0_range = np.arange(eb_start, eb_end + 0.001, eb_step)

            # Carregar ou gerar imagem de referência
            if self.image_path:
                img = Image.open(self.image_path).convert('L')
            else:
                grad = np.linspace(0, 255, 256, dtype=np.uint8)
                img = Image.fromarray(np.tile(grad, (256, 1)).astype(np.uint8))

            # Calcular curvas de BER teórica
            ber_uncoded = isdbt_tool.theoretical_ber(modulation, ebn0_range)
            gain_map = {0.5:3.0, 2/3:5.0, 3/4:6.0, 5/6:7.5, 7/8:8.5}
            gain_db = gain_map.get(coding_rate, 0.0)
            ber_coded = isdbt_tool.theoretical_ber(modulation, ebn0_range + gain_db)

            # Janela de plotagem da curva de BER
            plot_window = tk.Toplevel(self)
            plot_window.title("Curva de BER")
            fig = Figure(figsize=(6, 4), dpi=100)
            ax = fig.add_subplot(111)
            # Plotar curvas com legenda em português
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

            # Exibir taxa de dados aproximada
            data_rate = isdbt_tool.compute_data_rate(modulation, coding_rate, isdbt_tool.GUARD_INTERVALS[guard])
            ttk.Label(plot_window, text=f'Taxa de dados aproximada: {data_rate:.2f} Mb/s').pack(pady=5)

            # Criar janela para comparação de imagens (apenas uma janela)
            img_window = tk.Toplevel(self)
            img_window.title("Comparação de imagens")
            # Frame para organizar imagens e rótulos
            frame = ttk.Frame(img_window)
            frame.pack(padx=10, pady=10)
            # Armazenar referências às imagens para evitar coleta de lixo
            self._photo_refs = []
            # Considerar primeiro e último valor de Eb/N0
            snr_list = [float(ebn0_range[0]), float(ebn0_range[-1])] if len(ebn0_range) > 1 else [float(ebn0_range[0])]
            for idx, snr in enumerate(snr_list):
                # Degrada a imagem para este Eb/N0
                degraded_img = isdbt_tool.degrade_image(img, modulation, snr)
                 # monta lado a lado em resolução nativa (sem alterar qualidade fonte)
                w, h = img.size
                combined = Image.new('L', (w * 2 + 10, h))
                combined.paste(img, (0, 0))
                combined.paste(degraded_img, (w + 10, 0))
                # >>> Reduz SOMENTE para exibição, mantendo proporção e sem upscaling
                display_img = ImageOps.contain(
                    combined,
                    (self.MAX_DISPLAY_W, self.MAX_DISPLAY_H),
                    method=Image.Resampling.LANCZOS
                )
                # Converter para PhotoImage
                photo = ImageTk.PhotoImage(display_img)
                self._photo_refs.append(photo)
                # Rótulo da imagem
                img_label = tk.Label(frame, image=photo)
                img_label.grid(row=0, column=idx, padx=5, pady=5)
                # Rótulo para o Eb/N0
                eb_label = ttk.Label(frame, text=f'Eb/N0 = {snr:.2f} dB')
                eb_label.grid(row=1, column=idx, pady=(0, 10))

        except Exception as e:
            # Exibir mensagem de erro em português
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
            "Licença: GPL v3\n"
            "Mais informações em: www.obellab.com.br"
        )
        messagebox.showinfo("Sobre", info)


if __name__ == '__main__':
    app = ISDBTGui()
    app.mainloop()
