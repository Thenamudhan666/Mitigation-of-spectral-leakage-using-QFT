"""
UROP demo module 2: what does it cost to build each window state as a circuit?

Two measurements per window and per register size m:

 1. GENERIC cost  - build the window state with Qiskit's generic state
    preparation, transpile to {cx, rz, sx, x}, count CX gates and depth.
    (Generic loading depends on 2^m amplitudes, so we expect it to be nearly
    shape-independent. This is the honest baseline.)

 2. STRUCTURE (Schmidt rank) - split the m qubits into two halves and compute
    the numerical Schmidt rank of the window state across that cut.
    A window built from K cosine terms has rank <= 2K-1 (angle-addition
    identity), so structured circuits should be cheap for few-term windows.
    Kaiser has no such structure (rank grows with m). This is the test of
    the hypothesis "cost grows with number of cosine terms".

Run:  python qft_window_cost_demo.py
Needs: numpy, qiskit, matplotlib
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy import i0
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation

BASIS = ["cx", "rz", "sx", "x"]
M_VALUES = [3, 4, 5, 6]          # generic synthesis grows ~2^m; keep small
RANK_M_VALUES = [4, 6, 8, 10]    # Schmidt rank is cheap (just an SVD)


def cos_sum(N, coeffs):
    n = np.arange(N)
    w = np.zeros(N)
    for k, a in enumerate(coeffs):
        w += ((-1) ** k) * a * np.cos(2 * np.pi * k * n / (N - 1))
    return w


def make_windows(m):
    N = 2 ** m
    n = np.arange(N)
    x = (n - (N - 1) / 2) / ((N - 1) / 2)
    w = {
        "Rectangular":     (np.ones(N), 1),
        "Cosine":          (np.cos(np.pi * x / 2), 2),
        "Kaiser (a=6)":    (i0(np.pi * 6 * np.sqrt(1 - x ** 2)) / i0(np.pi * 6), None),
        "Hann":            (cos_sum(N, [0.5, 0.5]), 2),
        "Hamming":         (cos_sum(N, [0.54, 0.46]), 2),
        "Blackman-Harris": (cos_sum(N, [0.35875, 0.48829, 0.14128, 0.01168]), 4),
        "Flat-top":        (cos_sum(N, [0.21557895, 0.41663158, 0.277263158,
                                        0.083578947, 0.006947368]), 5),
    }
    # (amplitudes, number of cosine terms)  ; negatives are legal amplitudes
    return {k: (v / np.linalg.norm(v), t) for k, (v, t) in w.items()}


def generic_cost(amps, m):
    qc = QuantumCircuit(m)
    qc.append(StatePreparation(amps.astype(complex)), range(m))
    t = transpile(qc, basis_gates=BASIS, optimization_level=1, seed_transpiler=7)
    ops = t.count_ops()
    return int(ops.get("cx", 0)), t.depth()


def schmidt_rank(amps, m, tol=1e-9):
    h = m // 2
    mat = amps.reshape(2 ** h, 2 ** (m - h))
    s = np.linalg.svd(mat, compute_uv=False)
    return int((s > tol * s[0]).sum())


def main():
    # ---------- 1. generic gate cost ----------
    print("=== Generic state-prep cost (CX count / depth) ===")
    header = f"{'Window':<18}" + "".join(f"{'m=' + str(m):>14}" for m in M_VALUES)
    print(header)
    print("-" * len(header))
    cx_table = {}
    for name in make_windows(M_VALUES[0]):
        row = []
        for m in M_VALUES:
            amps, _ = make_windows(m)[name]
            cx, d = generic_cost(amps, m)
            row.append((cx, d))
        cx_table[name] = row
        print(f"{name:<18}" + "".join(f"{f'{cx}/{d}':>14}" for cx, d in row))

    # ---------- 2. Schmidt rank ----------
    print("\n=== Schmidt rank across the middle cut (lower = more structure) ===")
    header = f"{'Window (cos terms)':<26}" + "".join(f"{'m=' + str(m):>8}" for m in RANK_M_VALUES)
    print(header)
    print("-" * len(header))
    rank_table = {}
    for name in make_windows(RANK_M_VALUES[0]):
        ranks = []
        for m in RANK_M_VALUES:
            amps, terms = make_windows(m)[name]
            ranks.append(schmidt_rank(amps, m))
        _, terms = make_windows(RANK_M_VALUES[0])[name]
        label = f"{name} ({terms if terms else '-'})"
        rank_table[name] = ranks
        print(f"{label:<26}" + "".join(f"{r:>8}" for r in ranks))
    print("\nMax possible rank at m=10 is 32.")

    # ---------- plots ----------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for name, row in cx_table.items():
        a1.plot(M_VALUES, [c for c, _ in row], marker="o", label=name)
    a1.set_xlabel("phase qubits m"); a1.set_ylabel("CX gates (generic synthesis)")
    a1.set_yscale("log"); a1.set_title("Generic state-prep cost"); a1.grid(alpha=0.3)
    a1.legend(fontsize=8)
    for name, r in rank_table.items():
        a2.plot(RANK_M_VALUES, r, marker="s", label=name)
    a2.set_xlabel("phase qubits m"); a2.set_ylabel("Schmidt rank (middle cut)")
    a2.set_title("Structure: rank vs register size"); a2.grid(alpha=0.3)
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("cost_comparison.png", dpi=150)
    print("Saved cost_comparison.png")


if __name__ == "__main__":
    main()
