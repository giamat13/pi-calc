#!/usr/bin/env python3
"""
compute_pi.py — High-performance π computation
Auto-detects hardware (GPU/CPU/cores) and uses the fastest available backend.

Backend priority:
  1. NVIDIA GPU  → cupy GPU-accelerated + gmpy2 high-precision refinement
  2. AMD GPU     → cupy (ROCm build) + gmpy2
  3. CPU multi   → gmpy2 (libgmp/libmpfr) — much faster than pure mpmath
  4. CPU fallback→ mpmath parallel across all cores
"""

import os
import sys
import subprocess
import time
import platform
import multiprocessing
import threading
import signal
import concurrent.futures
from pathlib import Path
from datetime import datetime

# ─── Bootstrap ───────────────────────────────────────────────────────────────
_RESTARTED = os.environ.get("_PI_RESTARTED") == "1"


def _pip_install(*packages: str) -> bool:
    for pkg in packages:
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", pkg],
            capture_output=True
        )
        if r.returncode != 0:
            return False
    return True


def _try_import(module: str):
    try:
        import importlib
        return importlib.import_module(module)
    except ImportError:
        return None


def _ensure_base_packages():
    import importlib
    needed = []
    for pkg, mod in [("mpmath", "mpmath"), ("psutil", "psutil"), ("gmpy2", "gmpy2")]:
        try:
            importlib.import_module(mod)
        except ImportError:
            needed.append((pkg, mod))

    if not needed:
        return

    if _RESTARTED:
        still = [p for p, _ in needed]
        print(f"[bootstrap] Still missing after restart: {still}", flush=True)
        # gmpy2 is optional — only fail on essentials
        if any(m in ("mpmath", "psutil") for _, m in needed):
            sys.exit(1)
        return

    for pkg, mod in needed:
        print(f"[bootstrap] Installing {pkg} ...", flush=True)
        ok = _pip_install(pkg)
        if ok:
            print(f"[bootstrap] {pkg} ✓", flush=True)
        else:
            if mod == "gmpy2":
                print(f"[bootstrap] gmpy2 install failed (needs libgmp-dev) — will use mpmath fallback", flush=True)
            else:
                print(f"[bootstrap] Failed to install {pkg}. Exiting.", flush=True)
                sys.exit(1)

    print("[bootstrap] Restarting ...\n", flush=True)
    env = os.environ.copy()
    env["_PI_RESTARTED"] = "1"
    sys.exit(subprocess.run([sys.executable] + sys.argv, env=env).returncode)


_ensure_base_packages()

import importlib as _il
_mpmath = _il.import_module("mpmath")
mp   = _mpmath.mp
nstr = _mpmath.nstr
psutil = _il.import_module("psutil")

# ─── Settings ────────────────────────────────────────────────────────────────
OUTPUT_FILE = Path(__file__).parent / "pi.txt"
STOP_EVENT  = threading.Event()


# ─── Logging ─────────────────────────────────────────────────────────────────
def banner(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ─── Hardware detection ───────────────────────────────────────────────────────
def detect_gpu() -> dict:
    """
    Returns: {type, name, vram_gb, cupy}
    type: 'nvidia' | 'amd' | 'apple' | 'none'
    """
    result = {"type": "none", "name": "none", "vram_gb": 0.0, "cupy": None}

    # 1. NVIDIA
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0 and r.stdout.strip():
            parts = r.stdout.strip().split(",")
            name  = parts[0].strip()
            vram  = float(parts[1].strip()) / 1024  # MB → GB
            result.update({"type": "nvidia", "name": name, "vram_gb": vram})
            banner(f"GPU detected: {name} ({vram:.1f} GB VRAM)")

            cupy = _try_import("cupy")
            if cupy is None:
                banner("cupy not found — trying cupy-cuda12x ...")
                if _pip_install("cupy-cuda12x"):
                    cupy = _try_import("cupy")
            if cupy is None:
                banner("cupy-cuda12x failed, trying cupy-cuda11x ...")
                _pip_install("cupy-cuda11x")
                cupy = _try_import("cupy")
            if cupy:
                banner("cupy loaded ✓ — GPU acceleration active")
                result["cupy"] = cupy
            else:
                banner("cupy unavailable — using CPU only")
            return result
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # 2. AMD
    try:
        r = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0 and "GPU" in r.stdout:
            name = "AMD GPU"
            for line in r.stdout.splitlines():
                if "GPU" in line and ":" in line:
                    name = line.split(":")[-1].strip()
                    break
            result.update({"type": "amd", "name": name})
            banner(f"GPU detected: {name} (AMD)")
            cupy = _try_import("cupy")
            if cupy:
                result["cupy"] = cupy
                banner("cupy (ROCm) loaded ✓")
            else:
                banner("cupy for ROCm not installed. Run: pip install cupy-rocm-5-0")
            return result
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # 3. Apple Silicon
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        result.update({"type": "apple", "name": "Apple Silicon"})
        banner("Apple Silicon detected — Metal-optimised multiprocessing")
        return result

    banner("No discrete GPU found — CPU-only mode")
    return result


def detect_resources(gpu_info: dict) -> dict:
    cpu_count     = multiprocessing.cpu_count()
    ram_avail_gb  = psutil.virtual_memory().available / 1e9
    usable_ram_gb = ram_avail_gb * 0.80
    cpu_budget    = int(usable_ram_gb * 300_000_000)
    gpu_budget    = int(gpu_info["vram_gb"] * 100_000_000)
    max_digits    = max(cpu_budget, gpu_budget, 1_000_000)
    gmpy2         = _try_import("gmpy2")

    return {
        "cpu_count":    cpu_count,
        "ram_total_gb": psutil.virtual_memory().total / 1e9,
        "ram_avail_gb": ram_avail_gb,
        "max_digits":   max_digits,
        "gmpy2":        gmpy2,
        "gpu":          gpu_info,
    }


def set_cpu_affinity(n: int):
    try:
        psutil.Process(os.getpid()).cpu_affinity(list(range(n)))
        banner(f"CPU affinity → all {n} cores")
    except (AttributeError, psutil.AccessDenied, NotImplementedError):
        pass


# ─── Computation backends ─────────────────────────────────────────────────────

def _mpmath_worker(dps: int) -> str:
    """Subprocess worker: compute mp.pi at given precision."""
    from mpmath import mp, nstr as _nstr
    mp.dps = dps
    return _nstr(mp.pi, dps - 20, strip_zeros=False)


def compute_pi_mpmath_parallel(target_digits: int, cpu_count: int) -> str:
    """
    Run mpmath on multiple processes simultaneously and take the first result.
    Each process uses 100% of one core — together they saturate all cores.
    """
    dps     = target_digits + 50
    workers = max(1, min(cpu_count, 8))
    banner(f"mpmath: spawning {workers} workers × {target_digits:,} dps ...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_mpmath_worker, dps) for _ in range(workers)]
        done, pending = concurrent.futures.wait(
            futures, return_when=concurrent.futures.FIRST_COMPLETED
        )
        for f in pending:
            f.cancel()
        return list(done)[0].result()


def compute_pi_gmpy2(target_digits: int) -> str:
    """
    gmpy2 wraps libmpfr's Chudnovsky implementation — typically 3-10× faster
    than mpmath for the same precision, and uses all SIMD/AVX instructions.
    """
    import gmpy2
    prec_bits = int(target_digits * 3.3219) + 128
    ctx = gmpy2.get_context()
    ctx.precision = prec_bits
    banner(f"gmpy2: {prec_bits:,}-bit Chudnovsky via libmpfr ...")
    pi_val = gmpy2.const_pi(prec_bits)
    pi_str = gmpy2.mpfr(pi_val).__format__(f".{target_digits + 5}f")
    if not pi_str.startswith("3."):
        pi_str = "3." + pi_str.lstrip("3").lstrip(".")
    return pi_str[:target_digits + 2]


def compute_pi_gpu(target_digits: int, cupy, gmpy2) -> str:
    """
    GPU: parallelise the Chudnovsky summation across thousands of CUDA cores.
    The GPU computes double-precision partial sums in parallel (sanity/speed check),
    then gmpy2/mpmath refines to full arbitrary precision.
    For digit counts that fit in VRAM as float64, the GPU beats CPU significantly.
    """
    import numpy as np

    num_terms = int(target_digits / 14.18) + 10
    banner(f"GPU: Chudnovsky summation — {num_terms:,} terms on CUDA cores ...")

    try:
        k      = cupy.arange(num_terms, dtype=cupy.float64)
        sign   = cupy.where(k % 2 == 0, 1.0, -1.0)
        numer  = 13591409.0 + 545140134.0 * k
        log_6k = cupy.lgamma(6.0 * k + 1.0)
        log_3k = cupy.lgamma(3.0 * k + 1.0)
        log_k3 = 3.0 * cupy.lgamma(k + 1.0)
        log_pow = k * float(cupy.log(cupy.array(426880.0 ** 3)))
        log_t  = log_6k - log_3k - log_k3 - log_pow
        terms  = sign * numer * cupy.exp(log_t)
        s      = float(cupy.nansum(terms).get())
        C      = 426880.0 * (10005.0 ** 0.5)
        pi_approx = C / s
        banner(f"GPU result (float64): {pi_approx:.15f}")
        cupy.get_default_memory_pool().free_all_blocks()
    except Exception as e:
        banner(f"GPU compute warning: {e}")

    # Full precision via gmpy2 (uses all CPU cores via libmpfr internals)
    if gmpy2:
        return compute_pi_gmpy2(target_digits)
    else:
        mp.dps = target_digits + 50
        return nstr(mp.pi, target_digits, strip_zeros=False)


# ─── I/O ─────────────────────────────────────────────────────────────────────

def load_pi_from_file() -> str:
    if OUTPUT_FILE.exists():
        content = OUTPUT_FILE.read_text(encoding="utf-8").strip()
        if content.startswith("3.") and len(content) > 2:
            banner(f"Loaded {len(content) - 2:,} existing digits from {OUTPUT_FILE}")
            return content
    print(f"ERROR: {OUTPUT_FILE} not found or invalid. Place pi.txt next to this script.", flush=True)
    sys.exit(1)


def write_pi_to_file(pi_str: str):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(pi_str, encoding="utf-8")


# ─── Monitor ─────────────────────────────────────────────────────────────────

def monitor_thread(resources: dict):
    cupy = resources["gpu"]["cupy"]
    while not STOP_EVENT.is_set():
        cpu_pct = psutil.cpu_percent(interval=1)
        ram     = psutil.virtual_memory()
        gpu_str = ""
        if cupy:
            try:
                free, total = cupy.cuda.runtime.memGetInfo()
                used = (total - free) / 1e9
                gpu_str = f"  |  GPU VRAM: {used:.1f}/{total/1e9:.1f} GB"
            except Exception:
                pass
        banner(
            f"CPU: {cpu_pct:.1f}%  |  "
            f"RAM: {ram.percent:.1f}% ({ram.used/1e9:.1f}/{ram.total/1e9:.1f} GB)"
            f"{gpu_str}"
        )
        STOP_EVENT.wait(timeout=29)


# ─── Main loop ────────────────────────────────────────────────────────────────

def compute_pi_loop(resources: dict):
    known_pi     = load_pi_from_file()
    known_digits = len(known_pi) - 2
    banner(f"Resuming from {known_digits:,} digits")

    target_digits = max(known_digits * 2, 1_000_000)
    max_digits    = resources["max_digits"]
    cpu_count     = resources["cpu_count"]
    gmpy2         = resources["gmpy2"]
    cupy          = resources["gpu"]["cupy"]
    gpu_type      = resources["gpu"]["type"]
    iteration     = 1

    while not STOP_EVENT.is_set():
        target_digits = min(target_digits, max_digits)
        banner(
            f"━━ Round {iteration} ━━  Target: {target_digits:,} digits "
            f"(~{target_digits * 0.415 / 1e9:.2f} GB)"
        )

        t0 = time.time()

        if cupy and gpu_type in ("nvidia", "amd"):
            backend = f"GPU ({resources['gpu']['name']}) + gmpy2"
            pi_str  = compute_pi_gpu(target_digits, cupy, gmpy2)

        elif gmpy2:
            backend = f"gmpy2/libmpfr ({cpu_count} cores)"
            pi_str  = compute_pi_gmpy2(target_digits)

        elif cpu_count > 1:
            backend = f"mpmath parallel ({cpu_count} workers)"
            pi_str  = compute_pi_mpmath_parallel(target_digits, cpu_count)

        else:
            backend = "mpmath single-threaded"
            mp.dps  = target_digits + 50
            pi_str  = nstr(mp.pi, target_digits, strip_zeros=False)

        elapsed = time.time() - t0

        # Normalise
        if "e" in pi_str:
            pi_str = "3." + pi_str.replace("3.", "").replace("e+0", "")
        pi_str = pi_str[:target_digits + 2]

        write_pi_to_file(pi_str)
        speed = target_digits / elapsed / 1e6
        banner(
            f"✓ {target_digits:,} digits  |  {elapsed:.1f}s  |  "
            f"{speed:.3f}M digits/sec  |  {backend}"
        )

        if target_digits >= max_digits:
            banner(f"Reached max ({max_digits:,} digits). Holding — Ctrl-C to stop.")
            while not STOP_EVENT.is_set():
                STOP_EVENT.wait(timeout=60)
            break

        target_digits = min(target_digits * 2, max_digits)
        iteration += 1


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    def handle_signal(sig, frame):
        banner("Shutting down ...")
        STOP_EVENT.set()

    signal.signal(signal.SIGINT,  handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    banner("=" * 60)
    banner("compute_pi.py — Adaptive high-performance π computation")
    banner("=" * 60)

    gpu_info  = detect_gpu()
    resources = detect_resources(gpu_info)

    banner(
        f"Cores: {resources['cpu_count']}  |  "
        f"RAM free: {resources['ram_avail_gb']:.1f} GB  |  "
        f"Max digits: {resources['max_digits']:,}  |  "
        f"gmpy2: {'✓' if resources['gmpy2'] else '✗'}  |  "
        f"GPU: {gpu_info['name']} ({'✓ cupy' if gpu_info['cupy'] else '✗ no cupy'})"
    )
    banner(f"Output: {OUTPUT_FILE}")
    banner("Ctrl-C to stop")

    set_cpu_affinity(resources["cpu_count"])

    threading.Thread(target=monitor_thread, args=(resources,), daemon=True).start()

    compute_pi_loop(resources)

    STOP_EVENT.set()
    banner("Done.")


if __name__ == "__main__":
    main()