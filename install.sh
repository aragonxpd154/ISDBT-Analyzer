#!/usr/bin/env bash
# Instalador do ISDB-T Analyzer para Linux (Ubuntu, Zorin, Mint, Debian, Fedora…).
# Instala num ambiente virtual próprio, sem mexer no Python do sistema, e cria
# o atalho no menu de aplicativos e o comando `isdbt-analyzer` no terminal.
#
# Uso:  ./install.sh            instala (pergunta antes de usar sudo)
#       ./install.sh -y         instala sem perguntar
#       ./install.sh --dev      instala em modo desenvolvimento (editável + pytest/ruff)
#       ./install.sh --sem-sudo não tenta instalar pacotes do sistema
set -euo pipefail

APP="isdbt-analyzer"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}"
PREFIX="$DATA_DIR/$APP"
VENV="$PREFIX/venv"
BIN_DIR="$HOME/.local/bin"
DESKTOP_DIR="$DATA_DIR/applications"
ICON_DIR="$DATA_DIR/icons/hicolor/512x512/apps"

YES=0; DEV=0; NOSUDO=0
for arg in "$@"; do
  case "$arg" in
    -y|--yes) YES=1 ;;
    --dev) DEV=1 ;;
    --sem-sudo|--no-sudo) NOSUDO=1 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "Opção desconhecida: $arg"; exit 2 ;;
  esac
done

c_ok=$'\e[32m'; c_warn=$'\e[33m'; c_err=$'\e[31m'; c_b=$'\e[1m'; c_0=$'\e[0m'
say()  { printf '%s\n' "${c_b}==>${c_0} $*"; }
ok()   { printf '%s\n' "  ${c_ok}✔${c_0} $*"; }
warn() { printf '%s\n' "  ${c_warn}!${c_0} $*"; }
die()  { printf '%s\n' "  ${c_err}✘ $*${c_0}"; exit 1; }
ask()  { [ "$YES" = 1 ] && return 0; read -r -p "  $1 [S/n] " r; [[ -z "$r" || "$r" =~ ^[sSyY] ]]; }

say "ISDB-T Analyzer — instalação em $PREFIX"

# ---------------------------------------------------------------- Python
PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null || die "python3 não encontrado. Instale com: sudo apt install python3"
"$PY" - <<'EOF' || die "É necessário Python 3.10 ou mais novo."
import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)
EOF
ok "$("$PY" --version)"

# ---------------------------------------------------------------- pacotes do sistema
need_pkgs=()
if command -v dpkg >/dev/null; then
  "$PY" -c "import venv, ensurepip" 2>/dev/null || need_pkgs+=(python3-venv python3-pip)
  dpkg -s libxcb-cursor0 >/dev/null 2>&1 || need_pkgs+=(libxcb-cursor0)   # exigido pelo Qt 6.5+ no X11
  command -v ffmpeg >/dev/null || need_pkgs+=(ffmpeg)                      # opcional (há FFmpeg embutido)
  if [ ${#need_pkgs[@]} -gt 0 ]; then
    warn "Pacotes do sistema recomendados: ${need_pkgs[*]}"
    if [ "$NOSUDO" = 0 ] && ask "Instalar agora com sudo apt?"; then
      sudo apt-get update -qq && sudo apt-get install -y "${need_pkgs[@]}"
      ok "pacotes instalados"
    else
      warn "Pulando. Se a janela não abrir, rode: sudo apt install ${need_pkgs[*]}"
    fi
  else
    ok "dependências do sistema presentes"
  fi
elif command -v dnf >/dev/null; then
  rpm -q xcb-util-cursor >/dev/null 2>&1 || warn "Se a janela não abrir: sudo dnf install xcb-util-cursor"
fi

# ---------------------------------------------------------------- ambiente virtual
say "Criando ambiente virtual"
mkdir -p "$PREFIX"
if [ -d "$VENV" ]; then
  ok "reaproveitando $VENV"
else
  "$PY" -m venv "$VENV" || die "Falha ao criar o venv (instale python3-venv)."
  ok "venv criado"
fi
"$VENV/bin/python" -m pip install --upgrade --quiet pip wheel

say "Instalando o ISDB-T Analyzer e as dependências (pode levar alguns minutos)"
if [ "$DEV" = 1 ]; then
  "$VENV/bin/python" -m pip install --quiet -e "$SRC_DIR[dev]"
else
  "$VENV/bin/python" -m pip install --quiet --upgrade "$SRC_DIR"
fi
ok "$("$VENV/bin/isdbt-analyzer" --version)"

# ---------------------------------------------------------------- comando e atalho
say "Criando comando e atalho no menu"
mkdir -p "$BIN_DIR" "$DESKTOP_DIR" "$ICON_DIR"
ln -sf "$VENV/bin/isdbt-analyzer" "$BIN_DIR/$APP"
ok "comando: $BIN_DIR/$APP"

cp "$SRC_DIR/src/isdbt_analyzer/resources/icon.png" "$ICON_DIR/$APP.png"
sed -e "s|@EXEC@|$VENV/bin/isdbt-analyzer|g" -e "s|@ICON@|$APP|g" \
    "$SRC_DIR/packaging/$APP.desktop" > "$DESKTOP_DIR/$APP.desktop"
chmod +x "$DESKTOP_DIR/$APP.desktop"
command -v update-desktop-database >/dev/null && update-desktop-database -q "$DESKTOP_DIR" || true
command -v gtk-update-icon-cache >/dev/null && gtk-update-icon-cache -q -t "$DATA_DIR/icons/hicolor" 2>/dev/null || true
ok "atalho: $DESKTOP_DIR/$APP.desktop"

case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) warn "$BIN_DIR não está no PATH. Adicione ao ~/.bashrc:  export PATH=\"\$HOME/.local/bin:\$PATH\"" ;;
esac

echo
say "${c_ok}Pronto!${c_0} Abra pelo menu (\"ISDB-T Analyzer\") ou rode: ${c_b}isdbt-analyzer${c_0}"
echo "    Linha de comando: isdbt-analyzer --help"
echo "    Desinstalar:      $SRC_DIR/uninstall.sh"
