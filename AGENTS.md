# AGENTS Guidelines

This repository does not currently provide coding or contribution guidelines. The following rules help maintainers and automated agents create consistent pull requests.

## Commit messages
- Write concise commit messages in the imperative mood (e.g. "Add feature" not "Added").
- Explain the main change briefly in the summary line.
- Additional context should be provided in the body if needed.

## Coding style
- Python code must follow **black** and **isort** rules as defined in `.pre-commit-config.yaml`.
- Run `pre-commit run --files <file>` on changed files before committing to format code and check YAML.

## Testing
- Run `pytest -q` after modifications and ensure tests pass.
- If tests fail or dependencies are missing due to environment limits, mention this in the PR testing section.

## Pull request description
- Summarize notable changes and reference relevant file locations with Markdown links.
- Include the result of running tests (pass/fail) in the PR body.
