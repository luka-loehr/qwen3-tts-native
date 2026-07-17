LATEXMK ?= latexmk
BUILD_DIR ?= build
MAIN := main

.PHONY: all pdf clean static-check release-check source-archive

all: pdf

pdf: static-check
	@mkdir -p "$(BUILD_DIR)"
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -file-line-error \
		-outdir="$(BUILD_DIR)" "$(MAIN).tex"

static-check:
	@! grep -R --line-number --include='*.tex' '/Users/' .
	@! grep -R --line-number --include='*.tex' '\\today' .
	@! grep -R --line-number --include='*.tex' '\\write18\|minted' .

release-check: static-check
	@if grep -R --line-number 'PENDING_EVIDENCE' data sections tables figures; then \
		echo 'Refusing release: validated evidence placeholders remain.' >&2; \
		exit 1; \
	fi
	@test -f data/native-runs.dat
	@test -f data/sglang-runs.dat

source-archive: release-check pdf
	@rm -rf "$(BUILD_DIR)/arxiv-source"
	@mkdir -p "$(BUILD_DIR)/arxiv-source"
	@cp main.tex references.bib "$(BUILD_DIR)/arxiv-source/"
	@cp -R sections tables figures data "$(BUILD_DIR)/arxiv-source/"
	@find "$(BUILD_DIR)/arxiv-source" -name README.md -delete
	@if test -f "$(BUILD_DIR)/$(MAIN).bbl"; then \
		cp "$(BUILD_DIR)/$(MAIN).bbl" "$(BUILD_DIR)/arxiv-source/$(MAIN).bbl"; \
	fi
	@tar -C "$(BUILD_DIR)/arxiv-source" -czf \
		"$(BUILD_DIR)/qwen3-tts-native-paper-arxiv-source.tar.gz" .

clean:
	@if command -v "$(LATEXMK)" >/dev/null 2>&1; then \
		"$(LATEXMK)" -C -outdir="$(BUILD_DIR)" "$(MAIN).tex"; \
	fi
	@rm -rf "$(BUILD_DIR)"
