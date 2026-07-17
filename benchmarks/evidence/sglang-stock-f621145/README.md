# Stock SGLang-Omni Provenance at `f621145`

This capture documents the pinned stock SGLang-Omni 0.1.0 comparator built on an
NVIDIA DGX Spark (ARM64 / GB10) from repository commit `f621145`.

The captured local image ID is:

```text
sha256:9930cc808840e9bac03577f664fda8b44735eb1e531c56fec8e9ad14c9eb41d2
```

The only source compatibility patch removes the unavailable ARM64
`torchcodec==0.11.1` packaging dependency. It does not alter scheduling, model
execution, codec decoding, transport, or streaming behavior. The exact applied
patch is included in `provenance/applied-packaging-patch.diff`.

`pip-check.txt` is preserved verbatim, including known dependency conflicts. It
is audit evidence, not a claim that the full upstream environment has a clean
`pip check`. Runtime package versions, base-image provenance, upstream Git state,
system packages, image configuration, and DGX Spark identity are captured beside
it.

This directory proves comparator provenance only. Final Native-versus-SGLang
performance measurements are stored as separate evidence sets after the
controlled single-subject runs complete.
