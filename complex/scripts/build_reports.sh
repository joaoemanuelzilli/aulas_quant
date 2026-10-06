#!/bin/bash

# SCRIPT DE BUILD - Compile todos os arquivos LaTeX para PDF
# ============================================================
#
# Estrutura final:
#   reports/tex   -> fontes .tex
#   reports/build -> arquivos temporários do LaTeX
#   reports/pdf   -> PDFs finais
#
# Uso: bash scripts/build_reports.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TEX_DIR="$BASE_DIR/reports/tex"
BUILD_DIR="$BASE_DIR/reports/build"
PDF_DIR="$BASE_DIR/reports/pdf"
# Uso: bash build_reports.sh
# Uso: bash scripts/build_reports.sh

mkdir -p "$TEX_DIR" "$BUILD_DIR" "$PDF_DIR"

echo "=================================================="
echo "  COMPILANDO RELATÓRIOS LaTeX"
echo "=================================================="

python3 "$BASE_DIR/scripts/generate_report_charts.py" > /dev/null 2>&1

if ! command -v pdflatex &> /dev/null; then
    echo "❌ ERRO: pdflatex não encontrado"
    echo "Instale com: sudo apt-get install texlive-latex-base texlive-fonts-recommended"
    exit 1
fi

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

reports=(
    "relatorio_1_tese.tex|Relatório 1: Tese Estratégica"
    "relatorio_2_modelo.tex|Relatório 2: Modelo Técnico"
    "relatorio_3_backtest.tex|Relatório 3: Metodologia de Backtest"
    "relatorio_4_resultados.tex|Relatório 4: Resultados e Análise"
)

for i in "${!reports[@]}"; do
    entry="${reports[$i]}"
    tex_file="${entry%%|*}"
    report_name="${entry##*|}"
    tex_path="$TEX_DIR/$tex_file"
    pdf_name="${tex_file%.tex}.pdf"
    build_pdf="$BUILD_DIR/$pdf_name"
    final_pdf="$PDF_DIR/$pdf_name"

    echo ""
    echo -e "${BLUE}[$((i + 1))/4] Compilando ${report_name}${NC}"

    if [ ! -f "$tex_path" ]; then
        echo "❌ ERRO: arquivo não encontrado em $tex_path"
        exit 1
    fi

    pdflatex -halt-on-error -interaction=nonstopmode -output-directory "$BUILD_DIR" "$tex_path" > /dev/null 2>&1
    pdflatex -halt-on-error -interaction=nonstopmode -output-directory "$BUILD_DIR" "$tex_path" > /dev/null 2>&1 || true

    if [ -f "$build_pdf" ]; then
        mv "$build_pdf" "$final_pdf"
        echo -e "${GREEN}✓ $pdf_name gerado com sucesso em reports/pdf${NC}"
    else
        echo "❌ Erro ao compilar $tex_file"
        exit 1
    fi
done

echo ""
echo -e "${BLUE}Limpando arquivos temporários...${NC}"
rm -f "$BUILD_DIR"/*.aux "$BUILD_DIR"/*.log "$BUILD_DIR"/*.out "$BUILD_DIR"/*.toc "$BUILD_DIR"/*.lof "$BUILD_DIR"/*.lot

echo ""
echo "=================================================="
echo -e "${GREEN}  BUILD COMPLETO!${NC}"
echo "=================================================="
echo ""
echo "Arquivos gerados em reports/pdf:"
ls -lh "$PDF_DIR"/relatorio_*.pdf 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo "Estrutura organizada:"
echo "  reports/tex   -> fontes LaTeX"
echo "  reports/build -> temporários"
echo "  reports/pdf   -> PDFs finais"
echo ""
