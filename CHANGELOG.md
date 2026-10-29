# Changelog

## 0.3.9 (2026-10-29)

- Documentation only; the code is unchanged apart from the version number.
- The README cites the arXiv paper ([arXiv:2610.33826](https://arxiv.org/abs/2610.33826)) and links it on arXiv and alphaXiv.
- The examples page adds the LightningDiT-B/1 run with `blocking=False`: FID-50K 2.65 with guidance and 8.83 without,
  against 3.22 / 10.12 for the official recipe and 2.31 / 7.56 when training waits for the agent.

## 0.3.8 (2026-10-24)

- Two new read-only commands (#7, #8, by @mvanhorn) and one fix for the ssh mailbox; the optimizer math, the agent prompt and the agent's tools are unchanged.
- `opal watch <run>` prints a run's events as they are written and returns when the run finishes. `--timeout 0`
  prints what is already there; a positive `--timeout` is idle seconds after the last new event. A run killed without
  `close()` (SIGTERM, SIGKILL, out of memory) never writes its end, so it needs `--timeout`; watch then exits with 1.
- `opal compare a b` prints two run folders, or two result.json files, side by side: metric, command, decision counts,
  cost, waiting, last training loss, and the decisions that differ, matched by the update they were asked at.
  A result.json whose `opal_run` is relative to where training ran is matched to the run folder beside it.
- The ssh mailbox lists its folder in `sh` rather than the remote login shell, so an empty mailbox on a host whose
  login shell is zsh (such as macOS) no longer fails with "no matches found: */".
- SiT-B/2 is also scored with the ADM evaluator, the one the SiT paper uses: FID-50K 35.70 (AdamW), 30.39 (Muon),
  23.55 (OPAL).
- CONTRIBUTING.md says how pull requests are reviewed and merged.
- The mailbox tests stop their brokers when each test ends; a left-over broker made a later test fail now and then.

## 0.3.7 (2026-10-22)

- Documentation and examples only, apart from one printed notice; the optimizer math, the agent prompt and the agent's tools are unchanged.
- The README and the examples page report each task's newest OPAL run (RAEv2-B 3.19, Goldfish 77.25, LightningDiT-B/1 2.31) and no longer list older-system numbers.
- The examples page has a section on training without waiting for the agent (`blocking=False`), with the RAEv2-B official, blocking and non-blocking runs.
- The README says why long runs suit OPAL, with measured waiting times.
- `blocking="auto"` is no longer documented or offered by the example scripts. Using it prints that it is experimental and untested on real training runs.

## 0.3.6 (2026-10-22)

- Fixes from a from-scratch test of the install and the agent prompt; the optimizer math, the agent prompt and the agent's tools are unchanged.
- `opal doctor` says when the machine has an NVIDIA GPU that the installed torch cannot use; the README explains how to install the torch build for the driver first.
- The ssh mailbox broker explains a "Permission denied" from a host that asks for a password, and the guide and skill show how to share one connection.
- `opal estimate` no longer offers a decision count that still waits more than 4x, and says when training ends before the first answer.
- The report records the training waiting time for user scripts.
- The privacy filter keeps numbers such as `/1000`.
- The PyPI page links to the repository, the guide and the blog post, and README links work outside GitHub.

## 0.3.5 (2026-10-21)

- Reject uneven DDP inputs on every rank through the optimizer's Joinable interface instead of hanging (#2).
- Update installation, LightningDiT results and run records, remove the overfit result, and document mailbox leases, checkpoint state and the unfreezing workaround.
- Exclude local opal_runs artifacts from the wheel and source distribution.

- Normalize logged NumPy metrics and allowlist checkpoint observations for safe plain and Accelerate resumes (B1).
- Send SSH lease arguments and large answers over stdin; failed answer retries no longer block new requests (B2).
- Clear and release claims after SSH failures; let a restarted broker reclaim its own lease with generation fencing (B3).
- Keep the remote lease helper compatible with Python 3.6+ and check python3 availability in mailbox doctor (B4).
- Reclaim the same broker's lease immediately after restart instead of waiting for the stale timeout (F1).
- Log mailbox lease backoff once per request (F2).
- Keep aggressive LR interpolation on-device with tensor-list weights and exact zero/one endpoints (F3).
- Normalize NumPy updater state without calling rejected container hooks; record dropped fields in events and HTML, including quiet runs (F4).

- Check updater state with a closed allowlist without invoking custom serialization hooks.
- Redact form-encoded bearers, access tokens, triple-quoted, adjacent and bytes literals in diagnostics.
- Serialize mailbox lease takeovers and fence stale renewals, releases and responses by generation (#1).
- Save pending losses, analysis reports and authorized metrics so checkpoint resumes preserve observations (#3).

- Preserve Accelerate's public optimizer state, parameter groups and defaults through a checkpoint adapter.
- Resolve conjugate and negative parameter views before DDP byte fingerprints.
- Accept weights-only-safe updater values and preserve the starting updater exemption after analysis edits.
- Redact credentials inside nested escaped JSON, exception reprs and adjacent string fragments through the shared filter.
- Restore legacy Accelerate raw optimizer checkpoints with one warning about missing OPAL programs and progress.
- Preserve ordinary model names, identifiers, hashes, base64 code and credential-free URLs during redaction.
- Bound noncontiguous parameter fingerprint copies by slicing before flattening.
- Keep pending and unreported decision costs unknown in summaries and HTML until a numeric charge arrives.

- Redact API keys and bearer tokens from backend diagnostics, decision records, prompts, logs and HTML reports.
- Reject raw optimizer parameter-set changes after wrapping, before step or zero_grad (#5).
- Reject GradScaler steps on wrappers combining multiple optimizers.
- Include OPAL programs, versions and update counts in Accelerate save_state/load_state checkpoints.
- Share one version source across package metadata, runtime, Codex client and provenance.
- Update wrap docstrings for cost-budget stopping and the Codex backend.
- Advance truncated loss-window indices and record omitted observation counts.
- Settle cancelled decisions with a bounded wait and include late charges in reports.
- Discard pending wake reasons when no further agent call can use them.
- Reject unsupported or cyclic agent updater state before installation and checkpointing.
- Compare ordered parameter bytes with SHA-256 in DDP consistency checks.

- Resolve repository documentation from test files for non-editable installs.
- Set tested direct dependency lower bounds and require Python 3.10 or newer.
- Make the mailbox cancellation test use an instance-level wait.
- Publish complete mailbox claims atomically so concurrent brokers cannot run the same request twice.
- Use file rendezvous in the DDP crash-resume test to avoid ephemeral-port races.
- Include docs, skills, examples and the changelog in the source distribution.
- Add MNIST --max-cost-usd and --no-summarize flags.
- Run CPU CI across Linux, macOS and supported Python versions, including direct dependency minima.

## 0.3.4 (2026-10-21)

- Stop agent calls when the remaining cost budget is below the estimated call cost (#6).
- Apply the default waiting cap to the remaining schedule after early decisions, excluding agent waits from training time (#4).

## 0.3.3 (2026-10-21)

First public release.

- `opal.wrap` turns a PyTorch optimizer into a reactive optimizer: a coding agent (Claude Code by default) reads the
  training run and rewrites the update rule and its own analysis while the model trains.
- Examples on CPU and one GPU, and the reproductions in the README (SiT-B/2, RAEv2-B, Goldfish, modded-nanogpt).
- `opal report`, `opal show`, `opal estimate`, `opal doctor`; the Hugging Face Trainer through `opal.hf.OpalTrainer`.
- An experimental Codex backend (`backend="codex"`).
