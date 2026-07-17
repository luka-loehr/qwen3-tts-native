# Published benchmark report

This directory contains the release-committed, human-readable benchmark PDF.
It is generated only from a validated schema-v1.2 production manifest with
exactly twelve accepted cells: Native and stock SGLang at B1, B3, and B6 in two
rounds, with at least 200 successful measured requests per cell.

The complete raw evidence tree, production manifest, checksum inventory, and
release receipts are distributed as assets of the
[`v0.1.0` GitHub release](https://github.com/luka-loehr/qwen3-tts-native/releases/tag/v0.1.0).
They remain outside Git because raw request, packet, server-log, and 100-ms
telemetry records are substantially larger than the reviewed report. The PDF
identifies the exact evidence-manifest SHA-256 used to generate it.

## v0.1.0 artifacts

| Artifact | SHA-256 |
| --- | --- |
| `qwen3-tts-native-vs-sglang-stock-dgx-spark-2026-07-17-428307c-report.pdf` | `ab027f9116ee7af94b6f300145603843dcda446ccfa96bbdc4d8a14b2995cabf` |
| Production evidence manifest | `1ecc13f96a07bd9642a93a810c1d4e6821b450cbc1a6679d0ef60552b5994682` |
| `research/paper/qwen3-tts-native-paper.pdf` | `a65b6c27fbbd2c39572c624a0cfb0ea4f994267f245c356381b5ac84b995773b` |

Two independent report renders from separately copied, checksum-verified
evidence trees produced the same report digest. Poppler rendered all 21 pages
with an empty font cache; `pdffonts` reported only embedded, subsetted Bitstream
Vera Sans fonts. All pages were visually inspected before publication.

Do not place exploratory, partial, failed, synthetic, or hand-edited results in
this directory. Report generation and validation are documented in
[`reports/README.md`](../README.md).
