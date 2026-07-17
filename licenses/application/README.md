# Application license

The application owner approved Apache License 2.0 for the complete
`qwen3-tts-native` source repository on 2026-07-17.

The authoritative application license text is the root-level `LICENSE` file.
Every Cargo package declares `license = "Apache-2.0"`, and release images use
the OCI expression `Apache-2.0`. The release-metadata build context must copy
the root license byte-for-byte to `APPLICATION-LICENSE.txt`; it must not use a
placeholder or a model-only attribution file.

The Qwen model is distributed under its own Apache License 2.0 attribution in
`licenses/model/`. Keeping both records makes the application grant and the
third-party model grant independently auditable even though they use the same
SPDX identifier.
