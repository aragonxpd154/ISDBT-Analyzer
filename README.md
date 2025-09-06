<h1 align="center">
  <img alt="ISDBT Analyzer" src="https://raw.githubusercontent.com/aragonxpd154/ISDBT-Analyzer/main/images/isdbt-analyzer-icon.png" width="240"/>
  <br>
</h1>

<h4 align="center">
Ferramenta desktop (Python/Tkinter) para previsão de desempenho do sistema ISDB-T:
gera curvas teóricas de BER×Eb/N0, estima taxa de dados e mostra comparação de imagem
transmitida sob ruído AWGN (QPSK/16-QAM/64-QAM).
</h4>

<p align="center">
  <img alt="Github top language" src="https://img.shields.io/github/languages/top/aragonxpd154/ISDBT-Analyzer">
  <img alt="Github language count" src="https://img.shields.io/github/languages/count/aragonxpd154/ISDBT-Analyzer">
  <img alt="Repository size" src="https://img.shields.io/github/repo-size/aragonxpd154/ISDBT-Analyzer">
  <img alt="Github last commit" src="https://img.shields.io/github/last-commit/aragonxpd154/ISDBT-Analyzer">
  <a href="https://github.com/aragonxpd154/ISDBT-Analyzer/issues">
    <img alt="Repository issues" src="https://img.shields.io/github/issues/aragonxpd154/ISDBT-Analyzer">
  </a>
  <img alt="Github license" src="https://img.shields.io/github/license/aragonxpd154/ISDBT-Analyzer">
</p>

<p align="center">
  <a href="#rocket-technologies">Technologies</a>&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
  <a href="#information_source-how-to-use">How To Use</a>&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
  <a href="#status">Development Status</a>&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
  <a href="#memo-license">License</a>
</p>

<p align="center">
  <!-- Exemplo: insira aqui screenshots ou GIF de demonstração -->
  <!-- <img alt="Demo" src="https://raw.githubusercontent.com/aragonxpd154/ISDBT-Analyzer/main/images/demo.gif"> -->
</p>

## :rocket: Technologies

This project was used with the following technologies:

- [Python 3.x](https://www.python.org/)
- [Tkinter (GUI nativa do Python)](https://docs.python.org/3/library/tkinter.html)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Pillow (PIL)](https://python-pillow.org/)
- [PyInstaller (empacotamento)](https://pyinstaller.org/)

## :information_source: How To Use

Clone o repositório e execute localmente:

```bash
# Clone este repositório
git clone https://github.com/aragonxpd154/ISDBT-Analyzer.git
cd ISDBT-Analyzer
```

💻 Development Status

O projeto está funcional e em evolução.
Próximos passos (sugestões):

Implementar curvas BER com FEC mais realistas (Viterbi/RS) em vez do ganho fixo.
Adicionar perfis de canal (Brasil A/B/C/D/E com multi-caminho).
Exportar relatórios em PDF/PNG a partir da GUI.

<p align="center"> <img alt="Ícone" src="https://raw.githubusercontent.com/aragonxpd154/ISDBT-Analyzer/main/images/isdbt-analyzer-icon.png" width="200"> </p>
:memo: License

Este projeto é distribuído sob GPL v3.0. Veja LICENSE para mais detalhes.
© ObelLab — https://www.obellab.com.br
