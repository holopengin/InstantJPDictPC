## Agent skills

### Issue tracker

Issues, specs, and triage state live as markdown files under `.scratch/` in this repo. See `docs/agents/issue-tracker.md`.

### Triage labels

Five canonical triage roles, using the default label strings. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one root `CONTEXT.md` plus `docs/adr/`. See `docs/agents/domain.md`.

### Pipeline parity (pipeline-sharing/01 + 02)

Pipeline invariant specs live under `docs/pipeline/` (index: `docs/pipeline/README.md`);
the shared conformance corpus (format, tolerances, cases) under
`accessibility_daemon/tests/conformance/`. Maintenance rule: **a parity bug
fix updates the spec and adds a conformance case.** (Closes ticket 01's
AGENTS.md follow-up; see ticket 02.)
