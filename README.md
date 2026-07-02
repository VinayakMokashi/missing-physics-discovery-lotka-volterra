# Rediscovering Lotka–Volterra with Universal Differential Equations

Learn the **missing physics** of a predator–prey system directly from noisy data,
then recover it as a **human-readable equation** — a compact, end-to-end
Scientific Machine Learning (SciML) pipeline in Julia.

This project fits a **Universal Differential Equation (UDE)**: it keeps the parts
of the dynamics we know (linear growth / decay) and replaces the unknown
interaction terms with a neural network. After training, **sparse symbolic
regression** distills that neural network back into a clean algebraic expression,
recovering the original Lotka–Volterra interaction terms.

> Julia port of `SciML Mini Project.ipynb`, packaged as a runnable script.

---

## The problem

The ground-truth system is the classic Lotka–Volterra model:

```
dx/dt =  α·x  −  β·x·y        (prey / resource)
dy/dt =  γ·x·y  −  δ·y        (predator / consumer)
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
| 4 | Sparse **symbolic regression** (LASSO over a polynomial basis) recovers the equation | `Convex`, `SCS`, `Symbolics` |
| 5 | Plug recovered coefficients into the mechanistic ODE and compare to truth | `OrdinaryDiffEq`, `Plots` |

The neural network uses a radial-basis (`exp(-x²)`) activation, which pairs well
with the smooth polynomial structure the interaction terms actually have.

---

## Quick start

There are two ways to run it.

### Option A — reuse the existing bootcamp environment (recommended, verified)

This is the environment this project was tested on: **Julia 1.6.7** with the
pinned SciML `bootcamp` environment (the same one the other bootcamp projects
use). Point Julia at it and run:

```bash
julia --project="d:/SciML/bootcamp" "lotka_volterra_ude.jl"
```

The script automatically enables the three packages that aren't already in that
environment — `Convex`, `SCS` (for the LASSO step) and `Symbolics` (promoted
from a transitive to a direct dependency). This is a **one-time** step: the
first run downloads/precompiles them (a few minutes), every run after is fast.

### Option B — standalone, reproducible environment (for a fresh GitHub clone)

For anyone cloning the repo who does *not* have the bootcamp environment. Uses
the `Project.toml` in this folder with a modern Julia (**1.10+ recommended**).
The first run resolves and precompiles the full stack, which takes a while:

```bash
# from inside the project folder
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. lotka_volterra_ude.jl
```

After instantiating, commit the generated `Manifest.toml` to pin exact versions.

> **Runtime:** training runs 5000 ADAM iterations followed by up to 5000 L-BFGS
> iterations, so expect a few minutes of compute on a laptop (on top of the
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
| `04_missing_term_reconstruction.png` | NN-recovered interaction term vs. the true one |
| `05_reconstruction_and_error.png` | Reconstruction + L2 error over time |
| `06_overall.png` | Combined overview panel |
| `07_actual_vs_learned.png` | Final mechanistic model (recovered coefficients) vs. truth |

### `results/`
| File | Contents |
|------|----------|
| `data_noisy.csv` | The exact noisy dataset used for training |
| `losses.csv` | Full loss history (per iteration) |
| `trained_parameters.csv` | Flattened trained neural-network weights |
| `sindy_coefficients.csv` | LASSO coefficients (`β₁`, `β₂`) for every basis term |
| `learned_equations.txt` | The recovered symbolic equations |

## Expected result

The symbolic regression recovers the `x·y` interaction coefficients close to the
true values (`β ≈ 0.9`, `γ ≈ 0.8`), and the mechanistic model rebuilt from those
coefficients tracks the true trajectory almost exactly — the missing physics has
been rediscovered from data.

---

## Repository layout

```
.
├── lotka_volterra_ude.jl     # the whole pipeline (start here)
├── SciML Mini Project.ipynb  # original notebook this was ported from
├── Project.toml              # dependencies for a reproducible environment
├── figures/                  # generated PNGs (committed as evidence)
├── results/                  # generated CSV/txt artefacts (committed as evidence)
├── README.md
├── LICENSE                   # MIT
└── .gitignore
```

## How the code maps to the notebook

The script preserves the notebook's code cell-for-cell (each section is labelled
`CELL N`), with one deliberate change: **cells 12 and 13 are swapped**, because
the notebook's cell 12 depends on `X̂` / `ts` that are first defined in cell 13.
In a linear script the definitions have to come first. Figure- and result-saving
calls were added; the numerical logic is otherwise identical.

---

## Publishing to GitHub

From inside this folder:

```bash
git init
git add .
git commit -m "Lotka–Volterra UDE + symbolic regression"
gh repo create lotka-volterra-ude --public --source=. --push
```

(or create the repo on github.com and `git remote add origin … && git push -u origin main`).

## License

MIT — see [LICENSE](LICENSE).
