#!/usr/bin/env bash
# Remove o ISDB-T Analyzer instalado por install.sh.
# Uso: ./uninstall.sh          (mantém suas preferências)
#      ./uninstall.sh --tudo   (remove também as preferências da janela)
set -euo pipefail
APP="isdbt-analyzer"
DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}"
rm -rf "$DATA_DIR/$APP"
rm -f "$HOME/.local/bin/$APP" "$DATA_DIR/applications/$APP.desktop" "$DATA_DIR/icons/hicolor/512x512/apps/$APP.png"
command -v update-desktop-database >/dev/null && update-desktop-database -q "$DATA_DIR/applications" || true
if [ "${1:-}" = "--tudo" ]; then
  rm -f "${XDG_CONFIG_HOME:-$HOME/.config}/ObelLab/ISDBT-Analyzer.conf"
fi
echo "ISDB-T Analyzer removido."
