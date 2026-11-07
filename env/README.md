# Environments (Phase 1)

Conda: /opt/homebrew/Caskroom/miniforge/base (conda 26.3.2). Load it with `source env/conda_hook.zsh`.
Always call the environment's own interpreter: `"$CONDA_PREFIX/bin/python"`.

| Env | Python | GRANDE worktree (detached) | Install |
|---|---|---|---|
| fedgrande-ref | 3.12.14 (conda-forge) | third_party/grande_ref_07f7278b @ 07f7278b30ab9ebbbdc544e5f9a91f1b5df7fecb | `pip install -c env/ref_historical_constraints.txt -r <wt>/requirements.txt`, then `pip install --no-deps -e <wt>` |
| fedgrande-dev | 3.12.14 (conda-forge) | third_party/grande_dev_db79c860 @ db79c86060f06c15f46c3bfb9207dc30c3cb91ae | `pip install -c env/dev_shared_constraints.txt -e "<wt>[examples]"` |

Recreate the worktrees:
`git clone --no-checkout https://github.com/s-marton/GRANDE.git third_party/GRANDE-repo`, then
`git -C third_party/GRANDE-repo worktree add --detach ../grande_<ref|dev>_<short> <full hash>`.

Exact package lists:
- `env/snapshots/<env>.pip-freeze.txt`
- `<env>.conda-explicit.txt`
- `<env>.conda-export.yml`
- `<env>.json`: importlib.metadata versions, commit, and platform.
