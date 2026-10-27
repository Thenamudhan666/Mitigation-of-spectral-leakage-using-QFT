"""
UROP demo module: spectral leakage in QFT-based phase estimation, with window states.

What it does
------------
Ideal (noise-free) statevector simulation of QPE on m phase qubits with a
window state prepared on the phase register instead of plain Hadamards.
For an eigenphase phi that is NOT exactly representable on the grid
(the "non-periodic signal" case), it computes the exact output probability
histogram and reports:
  - success probability  (mass within +/- 1/2^m of phi)
  - leakage              (mass farther than 4 bins away)
  - peak sidelobe level  (dB relative to main peak)

Windows: rectangular, sine, cosine, Kaiser (baselines)
         Hann, Hamming, Blackman-Harris, flat-top (new to quantum)

Run:  python qft_window_leakage_demo.py
Needs: numpy, matplotlib
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy import i0  # modified Bessel I0

M = 6                      # phase qubits -> N = 64 bins
N = 2 ** M
PHI = (19 + 0.5) / N        # eigenphase exactly halfway between bins 19 and 20 (worst case)


def cos_sum(coeffs):
    """Generalised cosine-sum window: sum_k (-1)^k a_k cos(2*pi*k*n/(N-1))."""
    n = np.arange(N)
    w = np.zeros(N)
    for k, a in enumerate(coeffs):
        w += ((-1) ** k) * a * np.cos(2 * np.pi * k * n / (N - 1))
    return w


def windows():
    n = np.arange(N)
    x = (n - (N - 1) / 2) / ((N - 1) / 2)          # -1..1
    w = {
        "Rectangular":     np.ones(N),
        "Sine":            np.sin(np.pi * (n + 1) / (N + 1)),
        "Cosine":          np.cos(np.pi * x / 2),
        "Kaiser (a=6)":    i0(np.pi * 6 * np.sqrt(1 - x ** 2)) / i0(np.pi * 6),
        "Hann":            cos_sum([0.5, 0.5]),
        "Hamming":         cos_sum([0.54, 0.46]),
        "Blackman-Harris": cos_sum([0.35875, 0.48829, 0.14128, 0.01168]),
        "Flat-top":        cos_sum([0.21557895, 0.41663158, 0.277263158,
                                    0.083578947, 0.006947368]),
    }
    return w


def qpe_distribution(window, phi):
    """Exact QPE output probabilities. Window amplitudes -> controlled-U^k kicks
    phase exp(2*pi*i*phi*k) -> inverse QFT -> |amplitude|^2."""
    amp = window / np.linalg.norm(window)          # normalised quantum state
    k = np.arange(N)
    signal = amp * np.exp(2j * np.pi * phi * k)    # windowed, time-limited signal
    out = np.fft.fft(signal) / np.sqrt(N)          # (inverse) QFT up to conj/ordering
    p = np.abs(out) ** 2
    return p / p.sum()


def metrics(p, phi):
    peak = int(np.argmax(p))
    # circular distance from each bin to the true phase (in bins)
    d = np.abs(((np.arange(N) - phi * N) + N / 2) % N - N / 2)
    success = p[d <= 1.0].sum()
    leakage = p[d > 4.0].sum()
    side = p[d > 4.0].max() / p[peak]
    return success, leakage, 10 * np.log10(side + 1e-300)


def main():
    ws = windows()
    results = {}
    print(f"m = {M} phase qubits (N = {N} bins), true phase = {PHI:.5f} "
          f"(= bin {PHI * N:.2f}, off-grid)\n")
    print(f"{'Window':<18}{'Success P':>11}{'Leakage (>4 bins)':>20}{'Peak sidelobe (dB)':>21}")
    print("-" * 70)
    for name, w in ws.items():
        p = qpe_distribution(w, PHI)
        s, l, sl = metrics(p, PHI)
        results[name] = p
        print(f"{name:<18}{s:>11.4f}{l:>20.2e}{sl:>21.1f}")

    # ---- plot ----
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
    for ax, (name, p) in zip(axes.ravel(), results.items()):
        ax.semilogy(np.arange(N), p + 1e-18, lw=1.3)
        ax.axvline(PHI * N, color="r", ls="--", lw=0.8)
        ax.set_title(name, fontsize=10)
        ax.set_ylim(1e-12, 1)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("QPE output bin")
    for ax in axes[:, 0]:
        ax.set_ylabel("Probability (log)")
    fig.suptitle(f"Spectral leakage in QPE/QFT output by window state "
                 f"(m={M}, true phase off-grid, red = true phase)")
    fig.tight_layout()
    fig.savefig("leakage_comparison.png", dpi=150)
    print("\nSaved leakage_comparison.png")


if __name__ == "__main__":
    main()
