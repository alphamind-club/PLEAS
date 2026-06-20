# C23 - Phase Prompt Template Inventory

This folder contains five phase prompt templates copied from the BioClaw
repository:

- `bioplease-data-lake.md`
- `bioplease-environment.md`
- `bioplease-index.md`
- `bioplease-learn.md`
- `bioplease-tool-catalog.md`

`phase_prompt_template_inventory.csv` records raw SHA-256 hashes and
line-ending-normalized SHA-256 hashes for each copied artifact and the
corresponding file in the cloned BioClaw repo:

`BioClaw/.claude/skills/<template>/SKILL.md`

Result:

- All five copied templates match the BioClaw `main` source after normalizing
  line endings.
- Raw file hashes differ because of line-ending differences.
- The BioPlease `Beta` branch cloned from GitHub does not contain a matching
  `.claude/skills/` template directory, so this package cannot prove that the
  templates are unchanged between BioPlease and BioClaw.

Paper-ready wording:

> The C23 artifacts verify that 5/5 phase prompt templates in the supporting
> package exactly match the BioClaw `main` repository copies.

Do not claim "unchanged from BioPlease to BioClaw" unless a BioPlease source
template set is added for comparison.
