# Rediscovering Lotka–Volterra with Universal Differential Equations

Learn the **missing physics** of a predator–prey system directly from noisy data,
then recover it as a **human-readable equation** — a compact, end-to-end
Scientific Machine Learning (SciML) pipeline in Julia.

This project fits a **Universal Differential Equation (UDE)**: it keeps the parts
of the dynamics we know (linear growth / decay) and replaces the unknown
interaction terms with a neural network. After training, **sparse symbolic
regression** distills that neural network back into a clean algebraic expression,
recovering the original Lotka–Volterra interaction terms.

---

## The problem

The ground-truth system is the classic Lotka–Volterra model:

```
dx/dt =  α·x  −  β·x·y        (prey)
dy/dt =  γ·x·y  −  δ·y        (predator)
```

with true parameters `α = 1.3, β = 0.9, γ = 0.8, δ = 1.8`.

We pretend we **only know the linear terms** (`α·x` and `−δ·y`) and must *learn*
the nonlinear interaction terms (`−β·x·y` and `+γ·x·y`) from noisy observations.

## The approach

| Stage | What happens | Key tools |
|------:|--------------|-----------|
| 1 | Generate noisy data from the true ODE | `OrdinaryDiffEq` |
| 2 | Build a UDE — known linear terms + a neural net `U(x,y)` for the unknown terms | `Lux`, `ComponentArrays` |
| 3 | Train `U` to fit the data: **ADAM** warm-up → **L-BFGS** polish | `Optimization`, `SciMLSensitivity`, `Zygote` |
| 4 | Sparse **symbolic regression** (LASSO over a polynomial basis) recovers the equation | `LinearAlgebra`, `ModelingToolkit` |
| 5 | Plug recovered coefficients into the mechanistic ODE and compare to truth | `OrdinaryDiffEq`, `Plots` |

The neural network uses a radial-basis (`exp(-x²)`) activation, which suits the
smooth structure of the interaction terms we are trying to recover.

---

## How the code is organized

Everything lives in a single, linear script, [`lotka_volterra_ude.jl`](lotka_volterra_ude.jl),
split into seven labelled sections:

1. **Ground-truth model & noisy data** — integrate the true Lotka–Volterra ODE
   and add mean-scaled Gaussian noise to create the training set.
2. **Neural network** — a 4-layer MLP with radial-basis activations that will
   stand in for the unknown interaction terms.
3. **Hybrid UDE, prediction & loss** — the right-hand side keeps the known linear
   terms and adds the network; `predict` integrates it (with adjoint
   sensitivities), and `loss` is the MSE against the noisy data.
4. **Training** — ADAM warm-up followed by an L-BFGS polish, via `Optimization.jl`
   with `Zygote` reverse-mode AD through the ODE solve.
5. **Analysis** — trained trajectory vs. ground truth, the network's recovered
   interaction term vs. the true one, and the reconstruction error.
6. **Sparse symbolic regression** — build a polynomial candidate library, then
   solve a LASSO (`min ‖Φβ − y‖² + λ‖β‖₁`) so that only the genuinely-active
   basis terms survive, yielding an interpretable equation.
7. **Rebuild & compare** — plug the recovered coefficients back into the
   mechanistic ODE and compare against the true system.

The LASSO in step 6 is solved with a small self-contained coordinate-descent
routine (no external convex-optimization dependency). An equivalent `Convex.jl` +
`SCS` formulation of the *same* objective is included, commented, for reference;
the two agree to solver tolerance.

---

## Quick start

The script is self-contained — the sparse-regression step needs no packages
beyond the core SciML stack — so it runs end to end on either environment below.

### Option A — the pinned bootcamp environment (Julia 1.6.7)

If you already have the SciML `bootcamp` environment, point Julia at it:

```bash
julia --project="d:/SciML/bootcamp" "lotka_volterra_ude.jl"
```

Verified end to end on this setup: training, all seven figures, and every result
file, in a few minutes.

### Option B — standalone, modern Julia (1.10+)

For a fresh clone that does not have the bootcamp environment. Uses the
`Project.toml` in this folder; the first run resolves and precompiles the stack
(slow the first time):

```bash
# from inside the project folder
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. lotka_volterra_ude.jl
```

After instantiating, commit the generated `Manifest.toml` to pin exact versions.

> **Runtime:** training runs 5000 ADAM iterations followed by up to 5000 L-BFGS
> iterations, so expect a few minutes of compute on a laptop (on top of any
> one-time package precompile). Progress prints every 50 iterations.

You can also run it interactively — `include("lotka_volterra_ude.jl")` from a
Julia REPL — in which case the plots also pop up in a window.

---

## Outputs

Everything is written **next to the script**, so the repo stays self-contained.

### `figures/`
| File | Contents |
|------|----------|
| `01_noisy_data.png` | Ground-truth trajectory + noisy training data |
| `02_training_losses.png` | Loss curve (ADAM vs. L-BFGS) |
| `03_ude_trajectory.png` | Trained UDE trajectory vs. ground truth |
| `04_missing_term_reconstruction.png` | Network-recovered interaction term vs. the true one |
| `05_reconstruction_and_error.png` | Reconstruction + L2 error over time |
| `06_overall.png` | Combined overview panel |
| `07_actual_vs_learned.png` | Mechanistic model from the recovered coefficients vs. truth |

### `results/`
| File | Contents |
|------|----------|
| `data_noisy.csv` | The exact noisy dataset used for training |
| `losses.csv` | Full loss history (per iteration) |
| `trained_parameters.csv` | Flattened trained neural-network weights |
| `sindy_coefficients.csv` | LASSO coefficients (`β₁`, `β₂`) for every basis term |
| `learned_equations.txt` | The recovered symbolic equations |

## Result

Out of the 15 candidate terms, the sparse regression keeps **only** the `x·y`
interaction — every other coefficient is thresholded to zero — recovering the
exact functional form of the missing physics:

```
y1(t) ~ -0.834 * u1*u2      (true term: -0.9 * x*y)
y2(t) ~  0.701 * u1*u2      (true term: +0.8 * x*y)
```

The **structure is recovered perfectly**; the magnitudes sit slightly below the
true values because the L1 (LASSO) penalty shrinks coefficients toward zero —
expected behaviour. Rebuilding the mechanistic Lotka–Volterra model from the
recovered coefficients reproduces the predator–prey oscillation with only a
modest offset from that shrinkage (`07_actual_vs_learned.png`): the missing
physics has been rediscovered from noisy data.

---

## Repository layout

```
.
├── lotka_volterra_ude.jl   # the whole pipeline (start here)
├── Project.toml            # dependencies for a reproducible environment
├── figures/                # generated PNGs (committed as evidence)
├── results/                # generated CSV/txt artefacts (committed as evidence)
├── README.md
├── LICENSE                 # MIT
└── .gitignore
```

## License

MIT — see [LICENSE](LICENSE).
