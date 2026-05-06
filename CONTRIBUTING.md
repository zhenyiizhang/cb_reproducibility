# Contributing

Use this repository as a shared downstream notebook workspace.

## Workflow

1. Pull the latest `main`.
2. Create a new branch for your notebook or figure update.
3. Add notebooks under `notebooks/<dataset>/`.
4. Open a pull request back into `main`.

Do not commit directly to `main`.

## Notebook Rules

- Use English only.
- Do not hardcode local absolute paths.
- Write outputs to `results/<notebook_name>/`.
- Keep reusable logic in `downstream_helpers/` when possible.
- If a notebook is plot-only, make sure its required inputs are copied into this repository.

## What Not To Commit

- Anything under `results/`
- Large raw local data files
- Temporary files and caches

## Reviewer-Oriented Notebooks

If a notebook is intended for reviewers, it should:

- run in the `cb_pipeline` environment
- avoid any dependency on external Desktop paths
- document its expected outputs clearly
