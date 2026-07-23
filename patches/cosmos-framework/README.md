# Cosmos Framework local patches

## Slurm CPU affinity

- Patch: `0001-fix-respect-Slurm-cpuset-for-CPU-affinity.patch`
- Upstream repository: `https://github.com/NVIDIA/cosmos-framework.git`
- Upstream base commit: `90cd348877c37b888942c988b631eb1611bf2950`
- Local commit: `b4795a9`
- Local branch: `local/slurm-cpuset-affinity`

This patch intersects the CPU affinity reported by NVML with the CPU set
available to the current Slurm/cgroup job before calling
`os.sched_setaffinity()`. If the two sets do not overlap, it falls back to the
job's allowed CPU set.

Apply it to a clean checkout of the recorded base commit:

```bash
git checkout 90cd348877c37b888942c988b631eb1611bf2950
git am /project/peilab/wam/cosmos3_cy/patches/cosmos-framework/0001-fix-respect-Slurm-cpuset-for-CPU-affinity.patch
```

Validate it from the outer repository:

```bash
external/cosmos/packages/cosmos3/.venv/bin/python tests/test_device_cpu_affinity.py
```

The patch does not apply cleanly to the newer Edge framework checkout because
`cosmos_framework/utils/distributed.py` has changed upstream. Port the same
affinity-intersection logic manually while preserving the newer distributed
initialization code.
