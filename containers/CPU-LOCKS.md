# Linux CPU environment locks

`esm.lock.txt` and `dnabert2.lock.txt` target Linux x86-64, CPython 3.11 only.
They export the existing reviewed `uv.lock` using the corresponding SDK extra,
then replace generic PyPI torch with the official `torch==2.2.2+cpu` wheel.
The root lock and native macOS environments remain unchanged. All other package
versions/hashes, including NumPy 1.26.4, remain exactly as exported from the root lock.

```sh
python containers/generate_cpu_locks.py --verify-upstream
```

The generator removes `nvidia-*` and `triton`, which are dependencies of the generic
Linux CUDA wheel, not this CPU wheel. It asserts the remaining lock contains all
six unconditional CPU torch dependencies: filelock, typing-extensions, sympy,
networkx, jinja2, and fsspec. Optional opt-einsum/optree extras are not enabled.
Installation uses pip `--require-hashes`, including the exact direct wheel URL/hash.

Official source: https://download.pytorch.org/whl/cpu/torch/

- Wheel: `torch-2.2.2+cpu-cp311-cp311-linux_x86_64.whl`.
- SHA-256: `90089cae572672fb449c8ff1dc1b29daaffa117bf97ede7463dcd2fd1b991e4c`.
- Wheel size: 186,779,302 bytes.
- METADATA SHA-256: `a9aa8209f788d630c6415e7fbdc0e8e94327ea2ac6e7f16b374f1da9e6e35a6e`.
- Checked 2026-09-17 using the official index hash plus ZIP METADATA via bounded
  HTTP range requests (less than 1.3 MB). No model weights or CUDA packages fetched.

The official index currently points to download-r2.pytorch.org. The lock uses the
corresponding download.pytorch.org endpoint, whose byte ranges and metadata were
verified; pip must verify the complete wheel hash during installation.

CI builds each environment on a separate runner, verifies CPU-only torch metadata
and imports, and compares archived scoring output through native Python, Podman
and Apptainer. Passing those checks does not claim public-model inference parity.

## Sequence environment

`sequence.lock.txt` exports the `sequence` extra from the same root `uv.lock`.
It adds pandas, pyarrow and h5py with hashes, without PyTorch, CUDA, model weights
or upstream assay files. Regenerate it with the same generator; no CPU wheel
substitution is necessary for this environment. Each release CI matrix includes
`sequence` alongside `core`, `esm` and `dnabert2`.

`containers/parity.py --environment sequence` preserves the archived MFASS check
and adds synthetic FLIP2 CSV, mRNABench parquet and DART HDF5 inputs. It runs the
new protocol evaluators, two train-only embedding probes, a private scalar adapter
and an imported-embedding check. All synthetic results remain partial/subset.
`--compare native.json container.json` requires matching fields and scope, with
absolute and relative numeric tolerance `1e-9`. These are implementation checks,
not benchmark evidence or public-model inference claims.
