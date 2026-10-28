# Histórico de versões

## 2.0.2 — vídeo com perda maciça de pacotes

- **Corrige "FFmpeg falhou … no decoder found for: avs2".** Abaixo do QEF, quase todos os pacotes se perdiam,
  inclusive a PAT e a PMT. A PMT corrompida fazia o FFmpeg tomar o vídeo por outro formato. Agora o simulador
  age como um receptor já sintonizado, que guarda a PAT e a PMT em cache. O decodificador H.264 é forçado e
  não aborta por excesso de erros.
- **Tela "SEM SINAL"**: com perda de 90% ou mais (ou nada decodificável), o receptor perde o sincronismo e
  mostra "SEM SINAL", como numa TV real, em vez de dar erro. A mensagem explica e sugere subir o C/N.
- A aba de vídeo limpa os números da simulação anterior ao começar uma nova.

## 2.0.1 — desempenho da interface

- **Interface não trava mais.** As constelações viraram um mapa de densidade com persistência, sem eixos
  numéricos redesenhados a cada quadro. Cada quadro caiu de ≈52 ms para ≈18 ms, e numa máquina de um núcleo o
  atraso máximo da janela caiu de ≈600 ms para menos de 30 ms.
- **Ritmo adaptativo:** se a máquina não acompanha, o pipeline reduz sozinho a taxa de quadros.
- **Sem piscar:** números e cores só são atualizados quando mudam (cerca de 4 vezes por segundo), e mudar
  só o C/N não reconstrói a tela inteira.
- **Abas pesadas sob demanda:** os gráficos de Desempenho e OFDM só redesenham quando estão visíveis. A BER
  sem FEC em canais com multipercurso ficou cerca de 3 vezes mais rápida.
- **Simulação de FEC com prioridade baixa**, BLAS com 1 thread e o primeiro resultado em menos de 1 s.
- **Correção grave no vídeo:** um PTS corrompido fazia o FFmpeg gerar quadros sem fim, ocupando 100% da CPU
  mesmo depois de fechar o programa. Agora a decodificação é limitada ao número de quadros do original, tem
  tempo máximo, roda com prioridade baixa, e todo FFmpeg é encerrado ao sair.

## 2.0.0 — reescrita para Linux

### Correções de cálculo (afetam os resultados da v1.x)

- **BER teórica errada.** A v1 usava Es/N0 no lugar de Eb/N0 dentro do argumento da função Q. As curvas
  saíam cerca de **3 dB otimistas no QPSK** e **6 a 8 dB otimistas no 16/64-QAM**. Exemplo: com Eb/N0 = 6 dB,
  o QPSK correto tem BER 2,4×10⁻³, e a v1 mostrava 3,3×10⁻⁵. A v2 usa a BER **exata** do QAM com código Gray,
  somando as probabilidades sobre as regiões de decisão. Os testes conferem essa BER com Monte Carlo e com a
  fórmula fechada do QPSK.
- **Imagem degradada nunca voltava correta.** O mapeamento Gray era aplicado de novo na recepção, em vez de
  ser invertido, e o Gray não era bidimensional. Mesmo com **60 dB**, 94–99 % dos bytes voltavam errados. O
  "curto-circuito para Eb/N0 ≥ 40 dB" escondia o problema. A v2 usa Gray por eixo (como o ISDB-T), e há teste
  de regressão para isso.
- **"BER codificada" era um deslocamento fixo** da curva sem FEC (3 a 8,5 dB). A v2 implementa o código
  convolucional K=7 (171/133) com os puncionamentos reais e o decodificador de Viterbi soft. As curvas foram
  medidas por Monte Carlo e o RS(204,188) entra na conta. O C/N para QEF fica a até ±0,3 dB dos valores de
  referência na maioria dos casos (veja o README).
- `eval()` na taxa de código foi removido.

### Novidades

- Interface nova em **Qt (PySide6)**, numa janela única com tema escuro ou claro, alta resolução e as
  preferências salvas.
- **Pipeline ao vivo**: diagrama de blocos TX→RX com a métrica de cada etapa e quatro constelações em tempo
  real (mapeado, após o canal, equalizado e decisão com erros em vermelho). Mostra também |H(f)|, o histograma
  de bytes errados por pacote no RS e os contadores de BER pré e pós-Viterbi, PER e MER.
- **Camadas hierárquicas A/B/C**, com segmentos, modulação, código e entrelaçamento I. A taxa é exata pela
  estrutura de quadro (ex.: 12 seg 64-QAM 3/4 GI 1/16 = 17,843 Mb/s).
- **Perfis de canal Brasil A–E** e **eco SFN** configurável. Ecos além do intervalo de guarda geram ISI.
- **Vídeo em Transport Stream**: H.264/AAC em MPEG-TS, perda de pacotes de 188 bytes conforme a PER (com o
  bit TEI) e decodificação real com o FFmpeg, para ver macroblocos e congelamentos. O .ts corrompido pode ser
  exportado.
- **Imagem** lado a lado: original, sem FEC e com Viterbi + RS.
- **Parâmetros OFDM**: Tu, Tg, Δf, banda, quadro, TSP/quadro, atraso do entrelaçador, distância máxima de
  SFN, mapa dos 13 segmentos e diagrama de guarda × ecos.
- Exportação de **relatório PDF**, gráficos PNG/SVG/PDF, **CSV** e configuração **JSON**.
- **Linha de comando** com os subcomandos `taxa`, `desempenho`, `curvas`, `relatorio`, `imagem`, `video` e
  `calibrar`.
- **Instalador** `install.sh`, que cria um venv próprio, o atalho no menu e o comando `isdbt-analyzer`.
  Há também o `uninstall.sh`.
- Testes automatizados (pytest) e integração contínua no GitHub Actions.

### Repositório

- Código em `src/isdbt_analyzer` (pacote Python instalável via `pyproject.toml`).
- Os executáveis `.exe` antigos e os vídeos de teste (≈310 MB) saíram da árvore. O app gera o próprio vídeo de
  teste, e os binários antigos podem ficar em *GitHub Releases*.

## 1.8.0 e anteriores

Versões para Windows (Tkinter + PyInstaller).
