# README Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a concise Chinese `README.md` that explains the repository purpose, workflow, structure, and common commands for new maintainers.

**Architecture:** Reuse the current repo guide and existing docs as the source of truth, then compress them into a top-level README focused on orientation and first actions. Keep the README operational rather than academic, and avoid introducing new claims not already supported by the repository.

**Tech Stack:** Markdown, Git

---

### Task 1: Draft and publish the repository README

**Files:**
- Create: `README.md`
- Create: `docs/superpowers/plans/2026-07-01-readme-plan.md`

- [ ] **Step 1: Review existing repository-facing documentation**

Run: `sed -n '1,240p' docs/repo_guide.md`
Expected: A Chinese guide describing repo purpose, main workflows, and key paths.

- [ ] **Step 2: Write the README**

Create `README.md` with:
- A short Chinese overview of the repository scope
- Two main workstreams (`Piper14 SFT` and `Cosmos3 FD`)
- A compact directory guide
- Environment assumptions
- Common commands for readiness, DCP conversion, training, and offline eval
- A notes section explaining that `reports/` is output and `external/cosmos/` is external dependency state

- [ ] **Step 3: Verify the README content is present**

Run: `sed -n '1,260p' README.md`
Expected: The full Chinese README with the sections above and no placeholders.

- [ ] **Step 4: Commit only the README-related changes**

Run:
```bash
git add README.md docs/superpowers/plans/2026-07-01-readme-plan.md
git commit -m "docs: add repository README"
```
Expected: A commit containing only the new README and the plan file.

- [ ] **Step 5: Push the commit**

Run: `GIT_SSH_COMMAND='ssh -o StrictHostKeyChecking=no -p 443' git push`
Expected: The new docs commit is uploaded to `origin/master`.
