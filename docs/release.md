# Release & Install

This document describes how release artifacts are packaged and how to install
`sca` (alias: `semantic-code`).

## Binary names

- **Primary command**: `sca`
- **Alias**: `semantic-code`

Both commands run the same binary and share configuration/state.

## Release artifacts

Each GitHub Release publishes OS/arch-specific artifacts plus checksums.

Naming convention:

- `sca-<version>-<target>.tar.gz` (macOS/Linux)
- `sca-<version>-<target>.zip` (Windows)
- `sca-<version>-<target>.sha256`

Targets follow Rust triples (e.g., `x86_64-apple-darwin`,
`aarch64-apple-darwin`, `x86_64-unknown-linux-gnu`, `x86_64-pc-windows-msvc`).

## Checksum verification

```bash
sha256sum -c sca-<version>-<target>.sha256
```

## Install methods

### 1. Install script (from GitHub Releases)

```bash
curl -fsSL https://github.com/Luiz-Frias/semantic-code-agx/releases/latest/download/install.sh | sh
```

Defaults:

- Installs to a user-writable bin directory (prefers `~/.local/bin`).
- Installs both `sca` and `semantic-code` by default.
- Verifies checksums before placing the binary.
- macOS/Linux only. On Windows, use Scoop (after the first release) or install from source with Cargo.

### 2. Scoop (Windows) (after first release)

```powershell
scoop bucket add semantic-code https://github.com/Luiz-Frias/semantic-code-agx
scoop install semantic-code
```

Upgrade:

```powershell
scoop update semantic-code
```

### 3. mise (GitHub backend) (after first release)

```bash
mise use -g github:Luiz-Frias/semantic-code-agx@latest
```

Pin a specific version:

```bash
mise use -g github:Luiz-Frias/semantic-code-agx@v<version>
```

### 4. Cargo (from source)

```bash
cargo install semantic-code-cli --locked
```

Upgrade:

```bash
cargo install semantic-code-cli --locked --force
```

### 5. Homebrew (tap) (not published yet)

Homebrew publishing is not set up yet. A formula template is available at
`packaging/homebrew/semantic-code.rb.template`.

Once a tap is published, the install command will look like:

```bash
brew install <tap>/semantic-code
```

### 6. Winget (Windows) (not published yet)

Winget publishing is not set up yet. A manifest template is available at
`packaging/winget/semantic-code.yaml`.

## Verification

Confirm the install and build metadata:

```bash
sca --version
sca info
```

## Troubleshooting

- If the command is not found, ensure your install directory is on `PATH`.
- If using a Homebrew tap, run `brew doctor` and confirm `brew --prefix` is in `PATH`.
- If using Cargo, ensure `$HOME/.cargo/bin` is in `PATH`.

## Maintainers

Release packaging templates live in:

- `scripts/install.sh`
- `scripts/release/package_artifacts.py`
- `packaging/homebrew/semantic-code.rb.template`
- `packaging/winget/semantic-code.yaml`
- `bucket/semantic-code.json`

Release workflow:

- Run `just release-prep <target-branch> [--version <semver>]` to update `Cargo.toml` and `CHANGELOG.md`.
- Tagging `v<version>` on `main` **HEAD** triggers `.github/workflows/release.yml`, which builds artifacts, publishes checksums, and uploads `install.sh`.
- The workflow also updates `bucket/semantic-code.json` so Scoop users can upgrade without waiting on a separate bucket repo update.

### Scoop bucket setup

This repo is also a Scoop bucket: `bucket/semantic-code.json`.

1) Ensure `bucket/semantic-code.json` exists on the default branch (`main`).
2) Ensure GitHub Actions is allowed to push to `main` (or adjust the workflow to open a PR instead).
3) Create a release by tagging `v<version>`; the release workflow updates `bucket/semantic-code.json` with the new `version`, `url`, and `hash`.

### Homebrew tap setup

Homebrew installs formulae from a separate “tap” repository (a GitHub repo whose name starts with `homebrew-`).

1) Create a tap repository:
   - Recommended: `Luiz-Frias/homebrew-tap` (tap name: `luiz-frias/tap`)
2) Create a fine-grained PAT and add it as a secret to this repo:
   - Secret name: `HOMEBREW_TAP_TOKEN`
   - Access: repository `Luiz-Frias/homebrew-tap`
   - Permissions: Contents (read/write)
3) Tag a release on `main` HEAD. The release workflow renders `Formula/semantic-code.rb` from
   `packaging/homebrew/semantic-code.rb.template` and pushes it to `Luiz-Frias/homebrew-tap`.

Update `0.0.0` and `REPLACE_WITH_SHA256` placeholders in the packaging templates when publishing to Homebrew/Winget.
