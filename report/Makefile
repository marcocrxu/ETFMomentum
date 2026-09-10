# 编译: make  (需 lualatex + ctex/fandol, 如 TinyTeX: /root/.TinyTeX/bin/x86_64-linux)
PDF = analysis_report.pdf
SRC = analysis_report.tex
FIGS = figs/nav.png figs/mom.png figs/hold_days.png figs/hold_pnl.png

all: $(PDF)

$(PDF): $(SRC) $(FIGS)
	lualatex -interaction=nonstopmode -halt-on-error $(SRC)
	lualatex -interaction=nonstopmode -halt-on-error $(SRC)

clean:
	rm -f *.aux *.log *.out *.toc

.PHONY: all clean
