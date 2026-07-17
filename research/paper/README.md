# Qwen3-TTS Native research paper

This directory contains the English LaTeX source for the research paper
describing Qwen3-TTS Native. The sole author is **Luka Löhr**.

The source deliberately uses the standard LaTeX `article` class. arXiv is a
distribution platform, not a publisher-specific typesetting template, and the
project does not claim that an unofficial `arxiv.sty` file is required or
endorsed. The typography is compact, single-spaced, and intentionally close to
the familiar preprint appearance while remaining a conventional PDFLaTeX
document.

## Publication state

This is a buildable scaffold. It is **not publication-ready** while
`PENDING_EVIDENCE` appears anywhere in the source. Quantitative claims, run
statistics, optional registry digests, artifact hashes, workload identity, and
evidence-manifest identity must come from the accepted production bundle. They
must never be copied from an exploratory run, rounded dashboard, or hand-edited
spreadsheet.

`make release-check` is designed to fail until the evidence placeholders have
been replaced and the two run-data files exist. A successful PDF build alone
does not make the paper releasable.

## Source layout

```text
research/paper/
  main.tex                         Paper entry point
  references.bib                   Verified primary literature
  sections/                        One English file per paper section
  data/evidence_placeholders.tex   Sole machine-replaceable TeX data boundary
  data/*.dat                       Final manifest-derived monochrome plot data
  tables/                          Table presentation only; no embedded results
  figures/system_architecture.tex  Vector architecture figure
  figures/performance_summary.tex  Vector benchmark-plot template
  figures/generated/               Reserved for reviewed vector supplements
  tools/finalize_evidence.py        Deterministic production-manifest finalizer
  tests/                            Finalizer contract tests
  Makefile                         Build, release gate, and source archive
```

The finalizer revalidates the complete schema-v1.2 production evidence bundle
through the report pipeline, atomically replaces
`data/evidence_placeholders.tex`, and creates `data/native-runs.dat` and
`data/sglang-runs.dat`. Presentation files under `tables/` and `figures/` must
not be rewritten with hand-entered measurements.

## Build

The preferred publication toolchain is TeX Live 2025 with `latexmk`, PDFLaTeX,
and BibTeX. The Makefile also supports Tectonic as a deterministic local-build
fallback when `latexmk` is unavailable:

```bash
cd research/paper
make pdf
```

The PDF is written to `build/main.pdf`. The `latexmk` path uses
`-interaction=nonstopmode`, `-halt-on-error`, and `-file-line-error`; the
Tectonic path retains its log and bibliography intermediates for the same
post-build inspection. A warning must not be resolved by deleting content or
suppressing evidence checks. Release source remains PDFLaTeX-compatible and
must still pass an arXiv-compatible clean archive rebuild.

Once the production manifest and all referenced evidence are available:

```bash
cd research/paper
make test
make finalize-evidence \
  MANIFEST=/absolute/path/to/validated-evidence-root/manifest.json
make release-check
make pdf
```

`finalize-evidence` refuses test fixtures, incomplete or extra run cells,
protocol mismatches, short Git commits, invalid digests, unsafe TeX identity
tokens, and evidence that fails the complete report validator. It writes no
measured value from a CLI option. A missing registry image or missing
digest-bound compressed size is rendered as `N/A`; local unpacked size is
never used as a substitute.

Before publication:

```bash
make release-check
make source-archive
```

The second command creates
`build/qwen3-tts-native-paper-arxiv-source.tar.gz`. It stages the contents of
this directory at the archive root, includes the generated `main.bbl` when
available, and excludes local build products and Markdown guidance. Always
unpack that archive into an empty directory, rebuild it there, and inspect the
resulting PDF before upload.

## Official arXiv compatibility record

This record was checked against official arXiv documentation on 2026-07-17:

- arXiv currently supports TeX Live 2023 and 2025, with 2025 the default, and
  supports PDFLaTeX. This source selects only PDFLaTeX-compatible packages from
  the standard TeX Live distribution. See [TeX Live at
  arXiv](https://info.arxiv.org/help/faq/texlive.html).
- arXiv compiles from the root of the submitted archive even when a top-level
  file originally lived in a subdirectory. The archive target therefore places
  `main.tex` at the archive root and preserves only relative include paths. See
  [Submit TeX/LaTeX](https://info.arxiv.org/help/submit_tex.html).
- PDFLaTeX figures may be PDF, PNG, or JPEG, and arXiv does not perform
  on-the-fly figure conversion. The paper's diagrams and plots are native
  black-and-white TikZ/PGFPlots vector graphics; any supplemental generated
  figure must be committed as PDF. No figure embeds JavaScript.
- The submission includes `references.bib`; the source archive also includes a
  matching `main.bbl` when the local build produced one. This follows arXiv's
  requirement to include the required `.bib` or matching `.bbl` inputs.
- The source uses neither `\today`, double-spaced referee mode, shell escape,
  `minted`, external fonts, hidden source files, absolute paths, nor
  on-the-fly conversions. Auxiliary files and the locally generated PDF are not
  placed in the submission archive.
- The author must inspect arXiv's generated PDF before completing submission.
  A local build and automated checks do not replace this required preview.
- A `.tar.gz` archive is an officially supported multi-file upload form. See
  [Creating tar and zip files for
  upload](https://info.arxiv.org/help/tar.html).
- Title, authors, and abstract must also be supplied in arXiv's submission
  form. The form's metadata fields accept ASCII input; the author name should
  therefore be entered as `Luka L\"ohr` rather than pasted as Unicode. See
  [Metadata for required and optional
  fields](https://info.arxiv.org/help/prep.html).

These checks establish source compatibility only. Category selection, license,
endorsement, author-account verification, metadata approval, and the final
submission action remain the author's responsibility.

## Final evidence replacement contract

The finalizer must perform all of the following in one reviewed change:

1. Validate the production manifest and every digest-bound evidence file.
2. Populate all 12 ordered run rows: Native and stock SGLang, B1/B3/B6, rounds
   1 and 2.
3. Preserve stock SGLang's unknown EOS classification; transport completion
   must not be relabeled as natural EOS.
4. Write unrounded source values to both `.dat` files. Rounding belongs only to
   the TeX presentation layer.
5. Populate both exact Git commits, local image IDs, optional OCI digests,
   model-artifact evidence hashes, workload SHA-256, model revision, and
   evidence-manifest SHA-256.
6. Change `\FinalEvidenceAvailablefalse` to
   `\FinalEvidenceAvailabletrue` only after all preceding steps succeed.
7. Generate the abstract, results, and conclusion summaries from fixed English
   templates and the same validated bundle; no translation, paraphrasing, or
   language-model step is permitted.
8. Build, render every PDF page to an image, inspect every page, and run the
   clean archive rebuild before publication.

The paper must remain in English. No publication script may translate or
paraphrase source text automatically during evidence insertion.
