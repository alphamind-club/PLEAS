---
description: BioPLEASE environment bootstrap and diagnostics guidance
when_to_use: Use when the task needs biomedical software bootstrap, environment diagnosis, or awareness of the bundled BioPLEASE environment assets.
---

# BioPLEASE Environment

The environment bundle lives at:
`G:/BioClaw/BioPlease_tools/bioplease_env`

Important files:
- bio_env.yml
- cli_tools_config.json
- environment.yml
- install_cli_tools.sh
- install_r_packages.R
- new_software_v005.sh
- r_packages.yml
- README.md
- setup_path.sh
- setup.sh

Rules:
- Treat these assets as explicit bootstrap helpers, not silent background mutations.
- Explain when an environment change is necessary.
- Prefer diagnostics first, then the smallest environment adjustment that unblocks progress.
