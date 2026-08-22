# Uploading this project to GitHub

Recommended repository name:

`v11-radia-radiation-studio`

## Before publishing

From the repository root:

```bash
python3 scripts/preflight_github.py
python3 scripts/run_tests.py
```

Review `LICENSE`, `CITATION.cff`, `README.md`, and
`THIRD_PARTY_NOTICES.md`. If you want your personal name to appear in GitHub's
citation box, replace the project-level author in `CITATION.cff` before
publishing.

Never commit passwords, access tokens, private keys, `.env` files, or
`.streamlit/secrets.toml`.

## Method A — GitHub website + Git

1. On GitHub, create a new empty repository.
2. Choose the repository name and Public/Private visibility.
3. Because this folder already contains a README, `.gitignore`, and license,
   **do not initialize the GitHub repository with additional README,
   `.gitignore`, or license files**.
4. In Terminal, `cd` to this folder and run:

```bash
git init -b main
git add .
git status
git commit -m "Initial release: v9.0.0"
git remote add origin https://github.com/YOUR_USERNAME/v11-radia-radiation-studio.git
git remote -v
git push -u origin main
```

If Git asks for author identity, configure the name/email you want attached to
commits. If email privacy matters, use the no-reply address shown in your
GitHub email settings.

## Method B — GitHub CLI

After authenticating GitHub CLI:

```bash
git init -b main
git add .
git commit -m "Initial release: v9.0.0"
gh repo create v11-radia-radiation-studio --public --source=. --remote=origin --push
```

Replace `--public` with `--private` if desired.

## After the first push

Check the **Actions** tab. The included CI workflow should compile the project
and execute all regression scripts without requiring a real RADIA binary.

Recommended repository description:

> RADIA-based insertion-device magnetostatics, relativistic single-electron tracking, and Lienard-Wiechert radiation analysis in Python/Streamlit.

Recommended topics:

`radia`, `undulator`, `wiggler`, `synchrotron-radiation`,
`accelerator-physics`, `magnetostatics`, `scientific-computing`, `streamlit`,
`python`.

## Create the V9 release

After the initial push:

```bash
git tag -a v9.0.0 -m "V11 RADIA Radiation Studio v9.0.0"
git push origin v9.0.0
```

Then open GitHub **Releases** → **Draft a new release**, select tag `v9.0.0`,
use title:

`V11 RADIA Radiation Studio v9.0.0`

and paste the contents of `RELEASE_NOTES_v9.md`.

GitHub automatically provides source ZIP and tar archives for releases.

## Optional repository settings

For a public research repository, consider enabling:

- Issues
- Dependabot alerts/security updates
- secret scanning / push protection when available
- a branch ruleset on `main` requiring the CI workflow before merge

The included `.github/dependabot.yml` requests monthly dependency-update pull
requests for pip and GitHub Actions.
