# Releasing housing-star

Releases are built and published by [`.github/workflows/release.yml`](.github/workflows/release.yml)
when a GitHub Release is published. It runs the full CI, builds the sdist and the wheel once,
uploads them to TestPyPI, waits for your approval, and uploads the same files to PyPI. Uploads
use **trusted publishing**: PyPI trusts this repository's workflow directly, so no API token or
password is stored in GitHub.

`housing-star` is pure Python, so one universal wheel (`py3-none-any`) serves every supported
Python and platform. No per-platform wheels are needed.

## One-off setup

Do these once, in this order. The names must match exactly: project `housing-star`, workflow
`release.yml`, environments `testpypi` and `pypi`.

### 1. GitHub

1. **Commit email.** In GitHub, Settings → Emails, tick *Keep my email addresses private* and
   *Block command line pushes that expose my email*. Copy the `…@users.noreply.github.com` address
   shown there and use it for this repository:

   ```bash
   git config user.email "<id>+<username>@users.noreply.github.com"
   ```

2. **Environments.** In the repository, Settings → Environments, create two:
   - `testpypi`: no rules needed.
   - `pypi`: add yourself under *Required reviewers*, and under *Deployment branches and tags*
     allow only tags matching `v*`. The PyPI job then waits for you to click *Approve*.

3. **Protect `main`** (Settings → Rules → Rulesets, or Branches): require the CI checks to pass
   before merging, so a release is always cut from a green commit.

### 2. PyPI and TestPyPI

TestPyPI is a separate site with separate accounts. Do each step on both
[pypi.org](https://pypi.org) and [test.pypi.org](https://test.pypi.org).

1. **Create an account** and turn on two-factor authentication (required to publish).
2. **Add a pending trusted publisher.** Account settings → Publishing → *Add a new pending
   publisher* → GitHub:

   | Field | PyPI | TestPyPI |
   |---|---|---|
   | PyPI project name | `housing-star` | `housing-star` |
   | Owner | your GitHub user or organisation | same |
   | Repository name | the repository's name | same |
   | Workflow name | `release.yml` | `release.yml` |
   | Environment name | `pypi` | `testpypi` |

   A *pending* publisher reserves nothing until the first upload, which creates the project and
   makes the publisher permanent. Upload soon after adding it, or someone else could take the name.

## Each release

1. **Set the version.** Make `version` in `pyproject.toml` the new version, and give its
   `CHANGELOG.md` heading the release date: `## <version> - <YYYY-MM-DD>` in place of
   `- Unreleased`. Commit and push to `main`, and wait for CI to
   pass.
2. **Publish a GitHub Release.** Releases → *Draft a new release* → choose a new tag `v<version>`
   (for example `v0.1.0`) on `main` → paste the CHANGELOG section as the notes → *Publish release*.
   The workflow refuses a tag that does not match `pyproject.toml`.
3. **Check TestPyPI.** When the TestPyPI job finishes, look at
   <https://test.pypi.org/p/housing-star>, and if you like install it in a clean environment:

   ```bash
   pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ \
       "housing-star==<version>"
   ```

4. **Approve.** In the workflow run, *Review deployments* → `pypi` → *Approve and deploy*.
5. **Check PyPI.** <https://pypi.org/p/housing-star>, then `pip install housing-star` in a clean
   environment.

## If something goes wrong

- **A version can be uploaded only once.** PyPI never accepts the same version number again, even
  after deleting it. If a published release is broken, fix it and release the next patch version
  (for example 0.1.1). On PyPI you can also *yank* the broken one, which stops new installs from
  choosing it without breaking anyone who has pinned it.
- **CI or the build fails.** Nothing has been uploaded. Delete the GitHub Release and its tag, fix,
  push, and publish the release again.
- **The PyPI job fails after TestPyPI succeeded.** Re-run the failed jobs from the workflow run.
  The TestPyPI step skips files it already has.
