# =============================================================================
#  Universal Differential Equations (UDE) + Symbolic Regression
#  Rediscovering the Lotka–Volterra predator–prey dynamics from noisy data
# -----------------------------------------------------------------------------
#  Julia port of "SciML Mini Project.ipynb".
#
#  Pipeline
#    1.  Generate noisy data from the true Lotka–Volterra ODE
#    2.  Fit a Universal ODE: keep the known linear growth/decay terms and
#        replace the unknown interaction terms with a neural network
#    3.  Train the network (ADAM warm-up  ->  L-BFGS polish)
#    4.  Recover the missing interaction terms with sparse symbolic regression
#        (LASSO over a polynomial basis, solved with Convex + SCS)
#    5.  Plug the recovered coefficients back into the mechanistic ODE and
#        compare against the ground truth
#
#  Outputs (written next to this script):
#     ./figures   —  all PNG figures
#     ./results   —  losses, trained parameters, recovered equations & data
#
#  How to run  (see README.md for the full guide):
#     julia --project="d:/SciML/bootcamp" "lotka_volterra_ude.jl"
# =============================================================================

# --- Headless plotting when run as a script (`julia file.jl`) ----------------
# Lets the GR backend save PNGs without opening a window. Interactive REPL use
# (`include("lotka_volterra_ude.jl")`) is left untouched so plots still display.
if !isinteractive()
    ENV["GKSwstype"] = "100"
end

# --- Make sure the extra packages are loadable in the active environment ------
# Convex + SCS power the LASSO symbolic-regression step and are the only packages
# not already in the bootcamp environment. `Pkg.add` here is a one-time, additive
# step; every run afterwards is fast. (We deliberately do NOT `using Symbolics`
# directly — see CELL 15 — so it does not need to be added.)
import Pkg
for pkg in ("Convex", "SCS")
    if Base.find_package(pkg) === nothing
        @info "Installing package: $pkg"
        Pkg.add(pkg)
    end
end

# =============================================================================
#  CELL 0 — Packages & a reproducible RNG
# =============================================================================
# SciML Tools
using OrdinaryDiffEq, ModelingToolkit, DataDrivenDiffEq, SciMLSensitivity, DataDrivenSparse
using Optimization, OptimizationOptimisers, OptimizationOptimJL, LineSearches

# Standard Libraries
using LinearAlgebra, Statistics

# External Libraries
using ComponentArrays, Lux, Zygote, Plots, StableRNGs
gr()

# Set a random seed for reproducible behaviour
rng = StableRNG(1111)

# Output folders live next to this script, so everything is saved "in the same
# folder" and the repo stays self-contained.
const FIG_DIR = joinpath(@__DIR__, "figures")
const RES_DIR = joinpath(@__DIR__, "results")
mkpath(FIG_DIR)
mkpath(RES_DIR)
println("Figures -> ", FIG_DIR)
println("Results -> ", RES_DIR)

# =============================================================================
#  CELL 1 — Ground-truth Lotka–Volterra model & noisy training data
# =============================================================================
function lotka!(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α * u[1] - β * u[2] * u[1]
    du[2] = γ * u[1] * u[2] - δ * u[2]
end

# Define the experimental parameter
tspan = (0.0, 5.0)
u0 = 5.0f0 * rand(rng, 2)
p_ = [1.3, 0.9, 0.8, 1.8]

prob = ODEProblem(lotka!, u0, tspan, p_)
solution = solve(prob, Tsit5(), abstol = 1e-12, reltol = 1e-12, saveat = 0.25)

# Add noise in terms of the mean
X = Array(solution)
t = solution.t

x̄ = mean(X, dims = 2)
noise_magnitude = 5e-3

Xₙ = X .+ (noise_magnitude * x̄) .* randn(rng, eltype(X), size(X))

fig_data = plot(solution, alpha = 0.75, color = :black, label = ["True Data" nothing])
scatter!(t, transpose(Xₙ), color = :red, label = ["Noisy Data" nothing])
savefig(fig_data, joinpath(FIG_DIR, "01_noisy_data.png"))

# Persist the exact dataset the model is trained on (reproducibility).
open(joinpath(RES_DIR, "data_noisy.csv"), "w") do io
    println(io, "t,x_noisy,y_noisy")
    for i in eachindex(t)
        println(io, t[i], ",", Xₙ[1, i], ",", Xₙ[2, i])
    end
end

# --- CELL 2 — inspect the (random) initial condition --------------------------
@show u0

# =============================================================================
#  CELL 3 — Neural network for the unknown interaction terms
# =============================================================================
rbf(x) = exp.(-(x .^ 2))

# Multilayer FeedForward
const U = Lux.Chain(Lux.Dense(2, 5, rbf), Lux.Dense(5, 5, rbf), Lux.Dense(5, 5, rbf),
    Lux.Dense(5, 2))
# Get the initial parameters and state variables of the model
p, st = Lux.setup(rng, U)
const _st = st

# =============================================================================
#  CELL 4 — Hybrid UDE dynamics (known physics + neural closure)
# =============================================================================
# Define the hybrid model
function ude_dynamics!(du, u, p, t, p_true)
    û = U(u, p, _st)[1] # Network prediction
    du[1] = p_true[1] * u[1] + û[1]
    du[2] = -p_true[4] * u[2] + û[2]
end

# Closure with the known parameter
nn_dynamics!(du, u, p, t) = ude_dynamics!(du, u, p, t, p_)
# Define the problem
prob_nn = ODEProblem(nn_dynamics!, Xₙ[:, 1], tspan, p)

# =============================================================================
#  CELL 5 — Forward prediction of the UDE
# =============================================================================
function predict(θ, X = Xₙ[:, 1], T = t)
    _prob = remake(prob_nn, u0 = X, tspan = (T[1], T[end]), p = θ)
    Array(solve(_prob, Vern7(), saveat = T,
        abstol = 1e-6, reltol = 1e-6,
        sensealg = QuadratureAdjoint(autojacvec = ReverseDiffVJP(true))))
end

# =============================================================================
#  CELL 6 — Loss (mean squared error against the noisy data)
# =============================================================================
function loss(θ)
    X̂ = predict(θ)
    mean(abs2, Xₙ .- X̂)
end

# =============================================================================
#  CELL 7 — Training callback (records the loss history)
# =============================================================================
losses = Float64[]

callback = function (state, l)
    push!(losses, l)
    if length(losses) % 50 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end

# =============================================================================
#  CELL 8 — Optimization problem (Zygote reverse-mode AD)
# =============================================================================
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p))

# =============================================================================
#  CELL 9 — Stage 1 training: ADAM (fast, robust warm-up)
# =============================================================================
# NOTE: the notebook wrote `Adam(eta = 1e-3)`. The `eta` keyword only exists in
# newer Optimisers.jl; the pinned bootcamp environment wants the positional
# learning rate `Adam(1e-3)` (identical 1e-3 step size).
res1 = Optimization.solve(
    optprob, OptimizationOptimisers.Adam(1e-3), callback = callback, maxiters = 5000)
println("Training loss after $(length(losses)) iterations: $(losses[end])")

# =============================================================================
#  CELL 10 — Stage 2 training: L-BFGS (high-accuracy polish)
# =============================================================================
optprob2 = Optimization.OptimizationProblem(optf, res1.u)
res2 = Optimization.solve(
    optprob2, LBFGS(linesearch = BackTracking()), callback = callback, maxiters = 5000)
println("Final training loss after $(length(losses)) iterations: $(losses[end])")

# Rename the best candidate
p_trained = res2.u

# Persist the loss curve and the trained parameters.
open(joinpath(RES_DIR, "losses.csv"), "w") do io
    println(io, "iteration,loss")
    for (i, l) in enumerate(losses)
        println(io, i, ",", l)
    end
end
open(joinpath(RES_DIR, "trained_parameters.csv"), "w") do io
    println(io, "value")
    for v in collect(p_trained)
        println(io, v)
    end
end

# =============================================================================
#  CELL 11 — Training-loss curve (ADAM vs L-BFGS)
# =============================================================================
# Plot the losses
pl_losses = plot(1:5000, losses[1:5000], yaxis = :log10, xaxis = :log10,
    xlabel = "Iterations", ylabel = "Loss", label = "ADAM", color = :blue)
plot!(5001:length(losses), losses[5001:end], yaxis = :log10, xaxis = :log10,
    xlabel = "Iterations", ylabel = "Loss", label = "LBFGS", color = :red)
savefig(pl_losses, joinpath(FIG_DIR, "02_training_losses.png"))

# =============================================================================
#  CELL 13 — Trained trajectory vs. ground truth
# -----------------------------------------------------------------------------
#  NOTE: in the notebook this cell (13) is executed BEFORE cell 12, because
#  cell 12 relies on `X̂` and `ts` defined here. The order is swapped in this
#  script so the dependencies resolve top-to-bottom.
# =============================================================================
## Analysis of the trained network
# Plot the data and the approximation
ts = first(solution.t):(mean(diff(solution.t)) / 2):last(solution.t)
X̂ = predict(p_trained, Xₙ[:, 1], ts)
# Trained on noisy data vs real solution
pl_trajectory = plot(ts, transpose(X̂), xlabel = "t", ylabel = "x(t), y(t)", color = :red,
    label = ["UDE Approximation" nothing])
scatter!(solution.t, transpose(Xₙ), color = :black, label = ["Ground truth" nothing])
savefig(pl_trajectory, joinpath(FIG_DIR, "03_ude_trajectory.png"))

# =============================================================================
#  CELL 12 — Recovered vs. true missing interaction term
# =============================================================================
# Ideal unknown interactions of the predictor
Ȳ = [-p_[2] * (X̂[1, :] .* X̂[2, :])'; p_[3] * (X̂[1, :] .* X̂[2, :])']
# Neural network guess
Ŷ = U(X̂, p_trained, st)[1]

pl_reconstruction = plot(ts, transpose(Ŷ), xlabel = "t", ylabel = "U(x,y)", color = :red,
    label = ["UDE Approximation" nothing])
plot!(ts, transpose(Ȳ), color = :black, label = ["True Interaction" nothing])
savefig(pl_reconstruction, joinpath(FIG_DIR, "04_missing_term_reconstruction.png"))

# =============================================================================
#  CELL 14 — Reconstruction error & combined overview
# =============================================================================
# Plot the error
pl_reconstruction_error = plot(ts, norm.(eachcol(Ȳ - Ŷ)), yaxis = :log, xlabel = "t",
    ylabel = "L2-Error", label = nothing, color = :red)
pl_missing = plot(pl_reconstruction, pl_reconstruction_error, layout = (2, 1))

pl_overall = plot(pl_trajectory, pl_missing)
savefig(pl_missing, joinpath(FIG_DIR, "05_reconstruction_and_error.png"))
savefig(pl_overall, joinpath(FIG_DIR, "06_overall.png"))

# =============================================================================
#  CELL 15 — Candidate library (as callable functions) for symbolic regression
# =============================================================================
# The notebook wrote `using Symbolics, LinearAlgebra` here. Both are already
# available: LinearAlgebra was loaded in CELL 0, and `@variables` (plus the whole
# Symbolics symbolic-expression machinery) is re-exported by ModelingToolkit,
# also loaded in CELL 0. We rely on that re-export — as the bootcamp's
# HH-SciML-Project does (`using ModelingToolkit: @variables`) — because a late,
# separate `using Symbolics` triggers a very slow recompilation cascade on the
# pinned Julia 1.6.7 stack.

@variables u1 u2
basis_funcs = [
    u -> 1.0,
    u -> u[1],
    u -> u[1]^2,
    u -> u[1]^3,
    u -> u[1]^4,
    u -> u[2],
    u -> u[1] * u[2],
    u -> u[1]^2 * u[2],
    u -> u[1]^3 * u[2],
    u -> u[2]^2,
    u -> u[2]^2 * u[1],
    u -> u[2]^2 * u[1]^2,
    u -> u[2]^3,
    u -> u[2]^3 * u[1],
    u -> u[2]^4
]

# =============================================================================
#  CELL 16 — Same library as symbolic expressions (for pretty-printing)
# =============================================================================
@variables u₁,u₂

basis_symbols = [
    1,
    u₁,
    u₁^2,
    u₁^3,
    u₁^4,
    u₂,
    u₁ * u₂,
    u₁^2 * u₂,
    u₁^3 * u₂,
    u₂^2,
    u₂^2 * u₁,
    u₂^2 * u₁^2,
    u₂^3,
    u₂^3 * u₁,
    u₂^4
]

# =============================================================================
#  CELL 17 — Assemble regression inputs (NN inputs) and targets (NN outputs)
# =============================================================================
# Step 1: Use existing X̂ (inputs) and Ŷ (predictions)
X̂ = Array(X̂)  # size (2, T)
Ŷ = Array(Ŷ)  # size (2, T)

T = size(X̂, 2)
inputs = [X̂[:, i] for i in 1:T]      # list of [u1, u2]
targets = [Ŷ[:, i] for i in 1:T]     # list of [ŷ1, ŷ2]

# =============================================================================
#  CELL 18 — (Notebook scratch) first design-matrix attempt; superseded below
# =============================================================================
# 4. Build design matrix and solve for β₁ and β₂
Φ = [f(u) for f in basis_funcs, u in inputs]'  # note the transpose

# =============================================================================
#  CELL 19 — Rebuild inputs/targets (kept for 1:1 correspondence with notebook)
# =============================================================================
T = size(X̂, 2)
inputs = [X̂[:, i] for i in 1:T]      # list of [u1, u2]
targets = [Ŷ[:, i] for i in 1:T]     # list of [ŷ1, ŷ2]

# =============================================================================
#  CELL 20 — Final design matrix Φ and regression targets y₁, y₂
# =============================================================================
Φ = [Float64(f(u)) for u in inputs, f in basis_funcs]
y₁ = reshape(Float64.([y[1] for y in targets]), :, 1)        # size (T × 1)
y₂ = reshape(Float64.([y[2] for y in targets]), :, 1)        # size (T × 1)

# =============================================================================
#  CELL 21 — Sparse regression via LASSO (Convex.jl + SCS solver)
# =============================================================================
using Convex, SCS

# Step 3: Solve Lasso for y₁
λ = 0.1  # Lasso regularization strength
β₁_var = Convex.Variable(size(Φ, 2), 1)
# β₁_var = Convex.Variable(size(Φ, 2))

prob₁ = minimize(sumsquares(Φ * β₁_var - y₁) + λ * norm(β₁_var, 1))

Convex.solve!(prob₁, SCS.Optimizer)
β₁ = evaluate(β₁_var)

# Step 4: Solve Lasso for y₂
β₂_var = Convex.Variable(size(Φ, 2), 1)
# β₂_var = Convex.Variable(size(Φ, 2))

prob₂ = minimize(sumsquares(Φ * β₂_var - y₂) + λ * norm(β₂_var, 1))

Convex.solve!(prob₂, SCS.Optimizer)
β₂ = evaluate(β₂_var)

# =============================================================================
#  CELL 22 — Threshold small coefficients & pretty-print the learned equations
# =============================================================================
using Printf
# Step 5: Thresholding + pretty printing
threshold = 0.1# adjust as needed

function format_expr(β, φs, threshold)
    terms = []
    for (coeff, term) in zip(β, φs)
        if abs(coeff) > threshold
            coeff_str = @sprintf("%f", coeff)
            term_str = string(term)
            push!(terms, coeff_str * "*" * term_str)
        end
    end
    return join(terms, " + ")
end

expr₁_str = format_expr(β₁, basis_symbols, threshold)
expr₂_str = format_expr(β₂, basis_symbols, threshold)

println("📘 Learned symbolic expression for NN output 1 (thresholded):")
println("y₁(t) ≈ ", expr₁_str)

println("\n📘 Learned symbolic expression for NN output 2 (thresholded):")
println("y₂(t) ≈ ", expr₂_str)

# Persist the recovered equations and the full coefficient table.
open(joinpath(RES_DIR, "learned_equations.txt"), "w") do io
    println(io, "Learned symbolic expression for NN output 1 (thresholded):")
    println(io, "y1(t) ~ ", expr₁_str)
    println(io)
    println(io, "Learned symbolic expression for NN output 2 (thresholded):")
    println(io, "y2(t) ~ ", expr₂_str)
end
open(joinpath(RES_DIR, "sindy_coefficients.csv"), "w") do io
    println(io, "basis,beta1,beta2")
    for i in eachindex(basis_symbols)
        println(io, string(basis_symbols[i]), ",", β₁[i], ",", β₂[i])
    end
end

# =============================================================================
#  CELL 23 — Simulate the recovered mechanistic model vs. the true model
# =============================================================================
function lotka_volterra!(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α * u[1] - β * u[2] * u[1]
    du[2] = γ * u[1] * u[2] - δ * u[2]
end

# Initial conditions and time span
tspan = (0.0, 5.0)  # correct for ODEProblem

u0 = [3.1461493970111687,
 1.5370475785612603]
# Actual parameters
params_actual = [1.3, 0.9, 0.8, 1.8]

# Plug in your learned coefficients here (from symbolic regression).
# The 2nd and 3rd entries are the interaction coefficients recovered by the
# LASSO above (the u₁·u₂ term of β₁ and β₂); α and δ are the known parameters.
params_learned = [1.3, 0.893595, 0.796093, 1.8]

# Solve actual model
prob_actual = ODEProblem(lotka_volterra!, u0, tspan, params_actual)
sol_actual = solve(prob_actual, Tsit5(), abstol = 1e-12, reltol = 1e-12,saveat = 0.025)

# Solve learned model
prob_learned = ODEProblem(lotka_volterra!, u0, tspan, params_learned)
sol_learned = solve(prob_learned, Tsit5(), abstol = 1e-12, reltol = 1e-12,saveat = 0.025)

# For reference, report the freshly-recovered u₁·u₂ interaction coefficients
# (index 7 of the basis). Ȳ[1] = -β·xy and Ȳ[2] = +γ·xy, so β ≈ -β₁[7], γ ≈ β₂[7].
println("\nRecovered interaction coefficients this run:")
println("  β (prey loss)      ≈ ", -β₁[7], "   (true 0.9)")
println("  γ (predator gain)  ≈ ", β₂[7], "   (true 0.8)")

# =============================================================================
#  CELL 24 — Final comparison plot: actual vs. symbolic-regression model
# =============================================================================
# Plot comparison
fig_compare = plot(sol_actual.t, sol_actual[1,:], label="Resource/Capital - Actual", lw=2)
plot!(sol_actual.t, sol_actual[2,:], label="Companies/Agents - Actual", lw=2)
plot!(sol_learned.t, sol_learned[1,:], label="Resource/Capital - Learned", ls=:dash, lw=2)
plot!(sol_learned.t, sol_learned[2,:], label="Companies/Agents - Learned", ls=:dash, lw=2)
xlabel!("Time")
ylabel!("Value")
title!("Actual vs Symbolic Regression Model")
savefig(fig_compare, joinpath(FIG_DIR, "07_actual_vs_learned.png"))

println("\n✅ Done. All figures saved to $(FIG_DIR)")
println("✅ All numeric artefacts saved to $(RES_DIR)")
