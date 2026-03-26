# omlx — Claude Code Instructions

This is a **fork-based contribution project**. The upstream repo is owned by jundot.
All work here produces PRs to upstream — never ship directly to upstream/main.

---

## Git Remotes

| Remote | Repo | Purpose |
|--------|------|---------|
| `origin` | buftar/omlx | Your fork — push branches here freely |
| `upstream` | jundot/omlx | Owner's repo — fetch only, never push |

## Branch Rules (non-negotiable)

- **Never commit to `main`** — it is a mirror of upstream/main only
- **One branch per feature/fix** — keep them independent so PRs can be reviewed and merged separately
- **Branch naming**: `feature/<short-kebab-description>` (e.g. `feature/kv-compaction`)
- **Always branch from a synced main**:

```bash
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
git checkout -b feature/your-feature
```

## Submitting a PR

1. Push your feature branch to origin: `git push origin feature/your-feature`
2. Open PR on GitHub: **buftar/omlx → jundot/omlx**, target branch `main`
3. Describe what changed and why (not just what)

---

## Contribution Standards (per upstream CONTRIBUTING.md)

Source: https://github.com/jundot/omlx/blob/main/docs/CONTRIBUTING.md

### License Header
Every new `.py` file must include this as the first line:
```python
# SPDX-License-Identifier: Apache-2.0
```

### Testing
- Run fast tests during development: `pytest -m "not slow"`
- Before submitting a PR, run tests for affected modules: `pytest tests/test_<module>.py -v`
- Test file convention: source file `omlx/<module>.py` → test file `tests/test_<module>.py`
- New code must include corresponding tests

### Development Setup
```bash
pip install -e ".[dev]"
```

---

## Project Structure (key areas)

```
omlx/
├── omlx/
│   ├── api/        # OpenAI/Anthropic adapters
│   ├── cache/      # KV cache (paged, prefix, SSD)
│   ├── engine/     # Inference engines
│   ├── mcp/        # Model Context Protocol
│   ├── models/     # Model wrappers
│   ├── scheduler.py
│   ├── engine_core.py
│   ├── paged_cache.py
│   └── server.py
├── packaging/      # macOS menubar app (PyObjC)
├── tests/
└── docs/
```

---

## Syncing Upstream Changes

Run this before starting any new feature branch:
```bash
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```

If upstream has moved ahead while you're mid-feature, rebase your branch:
```bash
git fetch upstream
git rebase upstream/main
```
