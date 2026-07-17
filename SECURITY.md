# Security Policy

Qwen3-TTS Native includes an HTTP parser and service, a Rust scheduler, native
C ABI boundaries, CUDA kernels, model artifacts, and a release container. A
security issue in any of these layers can affect host availability, GPU memory,
request privacy, or supply-chain integrity.

## Supported versions

Security fixes are developed against the latest `main` commit and, after the
first release, the latest published release line.

| Version | Security support |
| --- | --- |
| `main` | Supported for current development and coordinated fixes. |
| Latest published release | Supported after the first release is published. |
| Older commits, branches, and unqualified image candidates | Not supported. |

At present, the first registry image is still in qualification. A local image,
development tag, branch build, or placeholder digest is not a supported
release.

## Report a vulnerability privately

Do not open a public issue, pull request, discussion, or benchmark report for a
suspected vulnerability.

Use one of these private channels:

1. Submit a private vulnerability report through
   [GitHub Security Advisories](https://github.com/luka-loehr/qwen3-tts-native/security/advisories/new).
2. If that channel is unavailable, email
   [luka@lukaloehr.com](mailto:luka@lukaloehr.com) with the subject
   `Qwen3-TTS Native security report`.

Include, where available:

- the affected source commit, release tag, and immutable image digest;
- the affected endpoint, ABI entry point, native component, or build step;
- the environment, GPU, driver, CUDA version, and runtime configuration;
- a minimal reproduction that does not contain credentials, model weights,
  personal data, or third-party voice samples;
- the expected behavior and observed security impact;
- relevant logs, stack traces, sanitizer output, or packet metadata after
  removing secrets and user content;
- whether the issue is already known to another party or has been disclosed.

Do not send passwords, registry tokens, private keys, production prompts,
generated private audio, or multi-gigabyte model artifacts. The maintainer will
request a safer transfer method if a larger private artifact is necessary.

## Response process

The project aims to:

- acknowledge a complete report within three business days;
- provide an initial severity and scope assessment within seven business days;
- coordinate validation, a fix, release timing, and disclosure with the
  reporter;
- credit the reporter when requested and legally possible.

These are response targets, not a service-level agreement. Complex native,
CUDA, driver, or upstream dependency issues may require additional time. Please
allow a coordinated fix and release before public disclosure.

## Security-relevant scope

Reports are particularly useful for:

- memory corruption, integer overflow, out-of-bounds access, use-after-free,
  double-free, race conditions, or unsafe ABI validation;
- malformed JSON, multipart, PCM, WAV, header, or request-ID handling that
  bypasses validation or causes unsafe behavior;
- unbounded CPU, host-memory, GPU-memory, queue, body, output, or shutdown
  resource consumption;
- cancellation, backpressure, capacity, or retirement flaws that permit
  cross-request interference or persistent denial of service;
- prompt, voice-description, generated-audio, metric, or request-identifier
  disclosure;
- unauthorized network exposure, authentication bypass in an integration, or
  unsafe default binding behavior;
- model/artifact path traversal, symlink attacks, hash or manifest bypasses,
  or loading material outside the pinned model contract;
- malicious or compromised Rust, CUDA, container, base-image, model, SBOM,
  provenance, signature, or release inputs;
- privilege escalation or escape from the documented hardened container run;
- discrepancies between published release claims and the exact registry
  digest or attestations.

Audio quality, pronunciation, voice preference, unsupported languages, and
ordinary model hallucination are not security vulnerabilities by themselves.
An upstream model, CUDA, driver, operating-system, or dependency vulnerability
may still affect this project; report it privately when the native runtime or
published image is exposed.

## Deployment security model

The server is an inference backend, not an internet edge. The standalone binary
binds to loopback by default. The production image listens on all container
interfaces and must be published only to host loopback or an explicitly private
service network. Neither form implements TLS termination or a concrete identity
provider. A production operator must:

- run only an accepted image by immutable digest, never an unqualified mutable
  tag;
- verify the image signature, provenance, SBOM, vulnerability report, model
  identity, and platform before deployment;
- keep the root filesystem read-only, run as UID/GID `10001`, drop all Linux
  capabilities, set `no-new-privileges`, and provide only the documented
  bounded `/tmp` tmpfs;
- expose only the required GPU device and keep the NVIDIA driver and container
  runtime patched;
- place the service behind TLS, authentication, authorization, rate limiting,
  request-size limits, request deadlines, and response-write timeouts;
- restrict `/metrics` and health endpoints to trusted infrastructure;
- avoid logging request bodies, voice descriptions, multipart PCM, or WAV
  output in proxies and observability systems;
- isolate tenants according to the deployment's privacy and threat model;
- monitor readiness tombstones, cancellations, capacity rejection, memory,
  process restarts, and GPU health;
- never mount alternate or writable model files over the embedded pinned
  artifact.

The server enforces bounded application limits, but those limits do not replace
edge controls. In particular, a slow client can occupy bounded response
capacity until the response is consumed or dropped; the deployment proxy must
enforce a response-write timeout.

## Supply-chain policy

Release images must satisfy
[`containers/RELEASE_CHECKLIST.md`](containers/RELEASE_CHECKLIST.md) for the
exact pushed digest. This includes pinned build inputs, model hashes, generated
third-party licenses, CycloneDX and BuildKit SBOMs, maximum provenance,
vulnerability scanning, signing, clean pulling, non-root execution, and real
GPU qualification.

Registry credentials and signing material must never be passed as Docker build
arguments, copied into a build context, committed to Git, or written into an
image layer. Suspected credential exposure must be revoked immediately even if
the affected commit or image is later deleted.

## Coordinated disclosure

After a fix and release are available, the project may publish a GitHub
Security Advisory describing the affected versions, severity, impact,
mitigation, and credit. The report and private correspondence will not be
published without considering the reporter's requested attribution and the
safety of users who still need to update.
