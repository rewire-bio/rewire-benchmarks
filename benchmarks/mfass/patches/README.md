# Third-party patches

## Licence exception for this directory

The repository is MIT-licensed (`LICENSE` at the root), and that remains the licence for
original Rewire code. **The files in this directory listed below are the exception.**

| File | Licence | Origin |
|---|---|---|
| `pangolin-5cf94b8-mask-per-gene-1.patch` | GPL-3.0 | Modifies Pangolin, https://github.com/tkzeng/Pangolin at `5cf94b8db938c658391b4305cd7ce33297d44ff7`. It contains upstream context lines. |
| `LICENSE-pangolin-GPL-3.0` | GPL-3.0 licence text | Exact bytes of upstream `LICENSE` at the same revision (git blob `f288702d2fa16d3cdf0035b15a9fcbc552cd88e7`, SHA-256 `3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986`) |

The patch, and any Pangolin tree it is applied to, are GPL-3.0 derivatives of Pangolin
(copyright its authors). They are distributed under GPL-3.0, whose text is in
`LICENSE-pangolin-GPL-3.0`, not under this repository's MIT licence.

Pangolin itself, including its model weights, is not vendored here. `mfass.pangolin_patch`
exports the pinned upstream commit from a local clone, applies the patch, and checks every
hash.

## `pangolin-5cf94b8-mask-per-gene-1.patch`

Upstream `process_variant` masks per-strand score arrays in place, so a gene's masked
scores depend on which same-strand genes were processed before it (upstream issue #29). The
patch gives each gene its own copy of the arrays and marks the installed version
`1.0.2+rewire.maskpergene1`. With `mask=False`, output is unchanged. The patch is equivalent
to unmerged upstream PR #30. See `docs/mfass-matched-study-registration.md`.
