"""
Mitigation of Spectral Leakage in the Quantum Fourier Transform (QFT)
for Non-Periodic Signal Analysis  --  consolidated research code
==================================================================

Part 1  Leakage simulation   : ideal statevector QPE/QFT with a window state
                               on the phase register (true phase off-grid).
Part 2  Gate cost            : generic state-prep CX count/depth in Qiskit.
Part 3  Structure            : Schmidt rank of each window state (proxy for
                               how cheap a structured circuit could be).

Windows: Rectangular, Sine, Cosine, Kaiser (alpha tuned)   -> baselines
         Hann, Hamming, Blackman-Harris, Flat-top          -> new to quantum

Limits (state these when presenting):
  * Ideal simulation only: no hardware, no noise model.
  * Part 2 uses GENERIC loading, which is shape-independent by construction.
  * Schmidt rank is a proxy for cost, not a measured gate count; for
    Blackman-Harris, Flat-top and Kaiser the numerical rank depends on the
    tolerance used.

Install:  pip install numpy matplotlib qiskit
Run:      python qft_window_research.py
Outputs:  leakage_comparison.png, cost_comparison.png, results printed
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy import i0
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation

# ----------------------------------------------------------------- settings
M_LEAK = 6                           # phase qubits for the leakage experiment
BIN_FRAC = 0.5                       # true phase halfway between two bins (worst case)
M_COST = [3, 4, 5, 6]                # generic synthesis grows ~2^m
M_RANK = [4, 6, 8, 10]
RANK_TOL = 1e-9
BASIS = ["cx", "rz", "sx", "x"]

# ------------------------------------------------------------------ windows
def cos_sum(N, coeffs):
    """Generalised cosine-sum window: sum_k (-1)^k a_k cos(2*pi*k*n/(N-1))."""
    n = np.arange(N)
    w = np.zeros(N)
    for k, a in enumerate(coeffs):
        w += ((-1) ** k) * a * np.cos(2 * np.pi * k * n / (N - 1))
    return w


def kaiser(N, alpha):
    n = np.arange(N)
    x = (n - (N - 1) / 2) / ((N - 1) / 2)
    return i0(np.pi * alpha * np.sqrt(1 - x ** 2)) / i0(np.pi * alpha)


def qpe_distribution(window, phi):
    """Exact QPE output probabilities for a window state on the phase register."""
    N = len(window)
    amp = window / np.linalg.norm(window)
    k = np.arange(N)
    out = np.fft.fft(amp * np.exp(2j * np.pi * phi * k)) / np.sqrt(N)
    p = np.abs(out) ** 2
    return p / p.sum()


def success_prob(p, phi):
    N = len(p)
    d = np.abs(((np.arange(N) - phi * N) + N / 2) % N - N / 2)
    return p[d <= 1.0].sum()


def tune_kaiser(m, phi_frac=BIN_FRAC):
    """Pick the Kaiser alpha that maximises success probability (grid search)."""
    N = 2 ** m
    phi = (N // 3 + phi_frac) / N
    alphas = np.linspace(0.0, 20.0, 201)
    scores = [success_prob(qpe_distribution(kaiser(N, a), phi), phi) for a in alphas]
    best = int(np.argmax(scores))
    return float(alphas[best]), float(scores[best])


def make_windows(m, alpha):
    N = 2 ** m
    n = np.arange(N)
    x = (n - (N - 1) / 2) / ((N - 1) / 2)
    raw = {
        "Rectangular":     (np.ones(N), 1),
        "Sine":            (np.sin(np.pi * (n + 1) / (N + 1)), None),
        "Cosine":          (np.cos(np.pi * x / 2), 2),
        f"Kaiser (a={alpha:g})": (kaiser(N, alpha), None),
        "Hann":            (cos_sum(N, [0.5, 0.5]), 2),
        "Hamming":         (cos_sum(N, [0.54, 0.46]), 2),
        "Blackman-Harris": (cos_sum(N, [0.35875, 0.48829, 0.14128, 0.01168]), 4),
        "Flat-top":        (cos_sum(N, [0.21557895, 0.41663158, 0.277263158,
                                        0.083578947, 0.006947368]), 5),
    }
    return {k: (v / np.linalg.norm(v), t) for k, (v, t) in raw.items()}


# ------------------------------------------------- Part 1: leakage simulation
def part1_leakage(alpha):
    m = M_LEAK
    N = 2 ** m
    phi = (N // 3 + BIN_FRAC) / N
    ws = make_windows(m, alpha)
    print(f"\n=== PART 1: leakage  (m={m}, N={N}, true phase bin {phi * N:.2f}) ===")
    print(f"{'Window':<18}{'Success P':>11}{'Leakage >4 bins':>18}{'Peak sidelobe dB':>19}")
    print("-" * 66)
    results = {}
    for name, (w, _) in ws.items():
        p = qpe_distribution(w, phi)
        d = np.abs(((np.arange(N) - phi * N) + N / 2) % N - N / 2)
        leak = p[d > 4.0].sum()
        side = 10 * np.log10(p[d > 4.0].max() / p.max() + 1e-300)
        results[name] = p
        print(f"{name:<18}{success_prob(p, phi):>11.4f}{leak:>18.2e}{side:>19.1f}")

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
    for ax, (name, p) in zip(axes.ravel(), results.items()):
        ax.semilogy(np.arange(N), p + 1e-18, lw=1.3)
        ax.axvline(phi * N, color="r", ls="--", lw=0.8)
        ax.set_title(name, fontsize=10)
        ax.set_ylim(1e-12, 1)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("QPE output bin")
    for ax in axes[:, 0]:
        ax.set_ylabel("Probability (log)")
    fig.suptitle(f"Leakage in QPE/QFT output by window state (m={m}, red = true phase)")
    fig.tight_layout()
    fig.savefig("leakage_comparison.png", dpi=150)
    print("Saved leakage_comparison.png")


# ------------------------------------------------------ Part 2: gate cost
def generic_cost(amps, m):
    qc = QuantumCircuit(m)
    qc.append(StatePreparation(amps.astype(complex)), range(m))
    t = transpile(qc, basis_gates=BASIS, optimization_level=1, seed_transpiler=7)
    return int(t.count_ops().get("cx", 0)), t.depth()


def part2_cost(alpha):
    print("\n=== PART 2: generic state-prep cost (CX / depth) ===")
    names = list(make_windows(M_COST[0], alpha))
    print(f"{'Window':<18}" + "".join(f"{'m=' + str(m):>13}" for m in M_COST))
    print("-" * (18 + 13 * len(M_COST)))
    table = {}
    for name in names:
        row = []
        for m in M_COST:
            amps, _ = make_windows(m, alpha)[name]
            row.append(generic_cost(amps, m))
        table[name] = row
        print(f"{name:<18}" + "".join(f"{f'{c}/{d}':>13}" for c, d in row))
    return table


# ------------------------------------------------- Part 3: Schmidt rank
def schmidt_rank(amps, m, tol=RANK_TOL):
    h = m // 2
    s = np.linalg.svd(amps.reshape(2 ** h, 2 ** (m - h)), compute_uv=False)
    return int((s > tol * s[0]).sum())


def part3_rank(alpha):
    print(f"\n=== PART 3: Schmidt rank, middle cut (tol={RANK_TOL:g}) ===")
    names = list(make_windows(M_RANK[0], alpha))
    print(f"{'Window (cos terms)':<28}" + "".join(f"{'m=' + str(m):>7}" for m in M_RANK))
    print("-" * (28 + 7 * len(M_RANK)))
    table = {}
    for name in names:
        ranks = [schmidt_rank(make_windows(m, alpha)[name][0], m) for m in M_RANK]
        terms = make_windows(M_RANK[0], alpha)[name][1]
        table[name] = ranks
        print(f"{f'{name} ({terms if terms else chr(45)})':<28}" +
              "".join(f"{r:>7}" for r in ranks))
    print("A K-term cosine window has rank <= 2K-1.  Rank for non-cosine-sum "
          "windows depends on the tolerance.")
    return table


def plot_costs(cost_table, rank_table):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for name, row in cost_table.items():
        a1.plot(M_COST, [c for c, _ in row], marker="o", label=name)
    a1.set(xlabel="phase qubits m", ylabel="CX gates (generic)",
           title="Generic state-prep cost", yscale="log")
    a1.grid(alpha=0.3); a1.legend(fontsize=8)
    for name, r in rank_table.items():
        a2.plot(M_RANK, r, marker="s", label=name)
    a2.set(xlabel="phase qubits m", ylabel="Schmidt rank (middle cut)",
           title="Structure vs register size")
    a2.grid(alpha=0.3); a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("cost_comparison.png", dpi=150)
    print("Saved cost_comparison.png")


# ------------------------------------------------------------------- main
def main():
    alpha, best = tune_kaiser(M_LEAK)
    print(f"Kaiser alpha tuned for m={M_LEAK}: alpha={alpha:g} (success P={best:.4f})")
    part1_leakage(alpha)
    cost = part2_cost(alpha)
    rank = part3_rank(alpha)
    plot_costs(cost, rank)


if __name__ == "__main__":
    main()
