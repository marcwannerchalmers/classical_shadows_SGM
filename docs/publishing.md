# Publish the documentation

The documentation site is built by Zensical and deployed from `main` with the
GitHub Actions workflow in `.github/workflows/docs.yml`. Pull requests build
and validate the site without publishing it. A deployment can also be started
manually from the Actions tab while viewing the `main` branch.

The canonical address is <https://docs.sgm-shadows.com/>.

## One-time GitHub setup

1. Push the repository, including the documentation workflow, to GitHub.
2. Open the repository's **Settings → Pages** page.
3. Under **Build and deployment**, choose **GitHub Actions** as the source.
4. In your personal GitHub **Settings → Pages**, add and verify
   `sgm-shadows.com`. GitHub will provide a TXT record; add that record at your
   DNS provider, complete verification, and leave the TXT record in place.
5. Return to the repository's **Settings → Pages** page. Enter
   `docs.sgm-shadows.com` under **Custom domain** and save it.

Domain verification is strongly recommended because it prevents another
GitHub account from claiming the domain for a Pages site.

## DNS setup

At the DNS provider for `sgm-shadows.com`, create this record:

| Type | Name | Value |
| --- | --- | --- |
| `CNAME` | `docs` | `marcwannerchalmers.github.io` |

The target must be the GitHub Pages user domain, without
`/classical_shadows_SGM` or another path. Do not use a wildcard DNS record for
this purpose.

DNS propagation may take up to 24 hours. Check the result with:

```bash
dig docs.sgm-shadows.com +short
```

The response should ultimately resolve through `marcwannerchalmers.github.io`.

## First deployment

Push a commit to `main`, then open the repository's **Actions** tab and select
the **Documentation** workflow. The `build` job validates and packages the
site; the `deploy` job publishes it to the `github-pages` environment.
If the workflow ran before Pages was enabled, use **Re-run all jobs** after
finishing the one-time setup above.

After GitHub's DNS check succeeds, return to **Settings → Pages** and enable
**Enforce HTTPS**. Then verify:

- <https://docs.sgm-shadows.com/>
- an internal page, such as
  <https://docs.sgm-shadows.com/getting-started/>
- the **Documentation** workflow has a successful deployment

## Updating the site

Every later push to `main` rebuilds and deploys the documentation. Pull
requests only run the validation build. To preview changes locally, run:

```bash
python -m pip install -r requirements-docs.txt
zensical serve
```

The generated `site/` directory is intentionally ignored by Git; GitHub Pages
receives it as a workflow artifact instead.
