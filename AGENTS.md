## Agent skills

### Issue tracker

Issues, specs, and triage state live as markdown files under `.scratch/` in this repo. See `docs/agents/issue-tracker.md`.

### Triage labels

Five canonical triage roles, using the default label strings. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one root `CONTEXT.md` plus `docs/adr/`. See `docs/agents/domain.md`.

### Shared OCR backend sync

`native/ppocr_ncnn/ppocr_ncnn_core.{h,cpp}`, `native/kana_size/kana_size_core.{h,cpp}`,
the model assets, and the shared pipeline data are mirrored from the Android
repo. Verify with `tools/sync_ppocr_core.sh --check /path/to/InstantJPDict`;
the `.githooks/pre-commit` hook and the `shared-ocr-sync` workflow run the same
check. The Android repo is the source of truth for those files; this repo is
the source of truth for `jpdict-core`.

### Pipeline parity (pipeline-sharing/01 + 02)

Pipeline invariant specs live under `docs/pipeline/` (index: `docs/pipeline/README.md`);
the shared conformance corpus (format, tolerances, cases) under
`tests/conformance/`. Maintenance rule: **a parity bug
fix updates the spec and adds a conformance case.** (Closes ticket 01's
AGENTS.md follow-up; see ticket 02.)
