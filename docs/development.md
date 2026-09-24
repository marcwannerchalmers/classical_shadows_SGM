# Documentation development

The site uses Zensical with a MkDocs-compatible `mkdocs.yml`. Griffe reads the
Python abstract syntax tree, and mkdocstrings renders the resulting API model.
Runtime imports are disabled in the handler configuration.

## Install the documentation tools

```bash
python -m pip install -r requirements-docs.txt
```

The documentation environment does not need the scientific runtime
dependencies because API extraction is static.

## Preview and build

```bash
zensical serve
zensical build --strict
```

Strict builds fail on invalid internal links or anchors. The CI workflow runs
the same build and separately asks Griffe to parse every referenced module with
Google-style docstrings and inspection disabled.

The repository check also verifies that every object and explicitly selected
member in the API reference has a docstring, that selected functions have
complete parameter and return annotations, and that fenced Python examples
parse with Python's AST parser:

```bash
python scripts/check_docs.py
```

## Writing API documentation

- Put accepted and returned types in annotations, not docstring prose.
- Use `Args:`, `Returns:`, `Raises:`, `Examples:`, and `Notes:` only where they
  add semantic information.
- Keep workflows, rationale, and long examples in Markdown pages.
- Add new public objects to [the API reference](reference/api.md).

## Checking API compatibility

Once releases are tagged, Griffe can compare a module against a previous Git
reference. For example:

```bash
griffe check shadow --search . --against v1.0.0 --format github
```

The project has several root-level public modules rather than one installable
package, so run the command once for each stability boundary that a release
promises. This check is intentionally not enforced until such a versioned API
contract exists.
