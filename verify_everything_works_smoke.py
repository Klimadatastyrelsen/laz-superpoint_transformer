#!/usr/bin/env python
"""Smoke test for a Superpoint Transformer environment.

Verifies that every core dependency imports and that CUDA is usable, printing a
machine-greppable ``SMOKE_*`` line per check so ``check_verification_logs.sh``
can confirm success/failure from the log file. Exits non-zero if anything fails.
"""

import importlib
import sys

# Run from the repo root so ``import src.*`` (e.g. the FRNN dependency) resolves.
_REPO_ROOT = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

results = []


def record(name, ok, detail=""):
    status = "OK" if ok else "FAIL"
    suffix = f" {detail}" if detail else ""
    print(f"SMOKE_{name}_{status}{suffix}", flush=True)
    results.append((name, ok))


def check_import(name, module, attr=None):
    try:
        mod = importlib.import_module(module)
        if attr is not None:
            getattr(mod, attr)
        version = getattr(mod, "__version__", "")
        record(name, True, f"({module} {version})".strip())
    except Exception as exc:  # noqa: BLE001 - we want to report any failure
        record(name, False, f"({module}: {type(exc).__name__}: {exc})")


def main():
    print("SMOKE_BEGIN", flush=True)

    check_import("TORCH", "torch")
    check_import("TORCHVISION", "torchvision")
    check_import("TORCHMETRICS", "torchmetrics")
    check_import("TORCH_GEOMETRIC", "torch_geometric")
    check_import("TORCH_SCATTER", "torch_scatter")
    check_import("TORCH_CLUSTER", "torch_cluster")
    check_import("PYG_LIB", "pyg_lib")
    check_import("PGEOF", "pgeof")
    check_import("GRID_GRAPH", "grid_graph", attr="edge_list_to_forward_star")
    check_import("CUT_PURSUIT", "pycut_pursuit.cp_d0_dist", attr="cp_d0_dist")
    check_import("FRNN", "src.dependencies.FRNN.frnn", attr="frnn_grid_points")
    check_import("PYTORCH_LIGHTNING", "pytorch_lightning")
    check_import("HYDRA", "hydra")
    check_import("LASPY", "laspy")
    check_import("OPEN3D", "open3d")

    # Functional CUDA check: device visible + a real kernel runs.
    try:
        import torch

        available = torch.cuda.is_available()
        record("CUDA_AVAILABLE", available,
                f"(device_count={torch.cuda.device_count()})")
        if available:
            name = torch.cuda.get_device_name(0)
            x = torch.randn(256, 256, device="cuda")
            y = (x @ x).sum().item()
            ok = y == y  # not NaN
            record("CUDA_MATMUL", ok, f"(gpu={name})")
        else:
            record("CUDA_MATMUL", False, "(no CUDA device)")
    except Exception as exc:  # noqa: BLE001
        record("CUDA_AVAILABLE", False, f"({type(exc).__name__}: {exc})")
        record("CUDA_MATMUL", False, "(skipped)")

    failed = [name for name, ok in results if not ok]
    if failed:
        print(f"SMOKE_FAILED ({len(failed)} checks failed: {', '.join(failed)})",
              flush=True)
        return 1
    print("SMOKE_ALL_OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
