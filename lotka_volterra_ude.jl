# =============================================================================
#  Universal Differential Equations (UDE) + Symbolic Regression
#  Rediscovering Lotka–Volterra predator–prey dynamics from noisy data
# -----------------------------------------------------------------------------
#  A hybrid scientific-machine-learning pipeline:
#
#    1. Generate noisy observations from the true Lotka–Volterra ODE.
#    2. Build a Universal ODE — keep the known linear growth/decay terms and
#       let a neural network stand in for the unknown interaction terms.
#    3. Train the network to fit the data (ADAM warm-up -> L-BFGS polish).
#    4. Recover the missing interaction terms as a sparse symbolic equation
#       (LASSO over a polynomial candidate library).
#    5. Rebuild the mechanistic ODE from the recovered coefficients and
#       compare it against the ground truth.
#
#  Everything is written next to this script:
#     ./figures   —  all PNG figures
#     ./results   —  loss history, trained parameters, recovered equations, data
#
#  Run:
#     julia --project="d:/SciML/bootcamp" "lotka_volterra_ude.jl"
#  (see README.md for environment details and a full walkthrough)
# =============================================================================

# Headless plotting: let the GR backend save PNGs without opening a window when
# run as a script. Interactive REPL use (`include(...)`) still shows the plots.
if !isinteractive()
    ENV["GKSwstype"] = "100"
end

# =============================================================================
#  Packages & a reproducible RNG
# =============================================================================
# SciML tools
using OrdinaryDiffEq, ModelingToolkit, DataDrivenDiffEq, SciMLSensitivity, DataDrivenSparse
using Optimization, OptimizationOptimisers, OptimizationOptimJL, LineSearches

# Standard libraries
using LinearAlgebra, Statistics, Printf

# External libraries
using ComponentArrays, Lux, Zygote, Plots, StableRNGs
gr()

# One fixed seed so every run (data noise + network init) is reproducible.
rng = StableRNG(1111)

# Output folders live next to this script, so the project stays self-contained.
const FIG_DIR = joinpath(@__DIR__, "figures")
const RES_DIR = joinpath(@__DIR__, "results")
mkpath(FIG_DIR)
mkpath(RES_DIR)
println("Figures -> ", FIG_DIR)
println("Results -> ", RES_DIR)

# =============================================================================
#  1. Ground-truth Lotka–Volterra model & noisy training data
# =============================================================================
function lotka!(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α * u[1] - β * u[2] * u[1]
    du[2] = γ * u[1] * u[2] - δ * u[2]
end

# Experimental setup: time span, initial condition, true parameters.
tspan = (0.0, 5.0)
u0 = 5.0f0 * rand(rng, 2)
p_ = [1.3, 0.9, 0.8, 1.8]

prob = ODEProblem(lotka!, u0, tspan, p_)
solution = solve(prob, Tsit5(), abstol = 1e-12, reltol = 1e-12, saveat = 0.25)

# Corrupt the clean solution with mean-scaled Gaussian noise.
X = Array(solution)
t = solution.t

x̄ = mean(X, dims = 2)
noise_magnitude = 5e-3
Xₙ = X .+ (noise_magnitude * x̄) .* randn(rng, eltype(X), size(X))

fig_data = plot(solution, alpha = 0.75, color = :black, label = ["True Data" nothing])
scatter!(t, transpose(Xₙ), color = :red, label = ["Noisy Data" nothing])
savefig(fig_data, joinpath(FIG_DIR, "01_noisy_data.png"))

# Persist the exact dataset the model is trained on.
open(joinpath(RES_DIR, "data_noisy.csv"), "w") do io
    println(io, "t,x_noisy,y_noisy")
    for i in eachindex(t)
        println(io, t[i], ",", Xₙ[1, i], ",", Xₙ[2, i])
    end
end

@show u0

# =============================================================================
#  2. Neural network for the unknown interaction terms
# =============================================================================
rbf(x) = exp.(-(x .^ 2))

# A small multilayer perceptron with radial-basis activations, which suit the
# smooth structure of the interaction terms we are trying to recover.
const U = Lux.Chain(Lux.Dense(2, 5, rbf), Lux.Dense(5, 5, rbf), Lux.Dense(5, 5, rbf),
    Lux.Dense(5, 2))
p, st = Lux.setup(rng, U)
const _st = st

# =============================================================================
#  3. Hybrid UDE dynamics, forward prediction & loss
# =============================================================================
# Known physics (linear growth/decay) + a neural closure for the missing terms.
function ude_dynamics!(du, u, p, t, p_true)
    û = U(u, p, _st)[1]                 # network prediction of the interaction
    du[1] = p_true[1] * u[1] + û[1]
    du[2] = -p_true[4] * u[2] + û[2]
end

# Close over the known parameters.
nn_dynamics!(du, u, p, t) = ude_dynamics!(du, u, p, t, p_)
prob_nn = ODEProblem(nn_dynamics!, Xₙ[:, 1], tspan, p)

# Solve the UDE forward for a given parameter vector.
function predict(θ, X = Xₙ[:, 1], T = t)
    _prob = remake(prob_nn, u0 = X, tspan = (T[1], T[end]), p = θ)
    Array(solve(_prob, Vern7(), saveat = T,
        abstol = 1e-6, reltol = 1e-6,
        sensealg = QuadratureAdjoint(autojacvec = ReverseDiffVJP(true))))
end

# Mean-squared error against the noisy observations.
function loss(θ)
    X̂ = predict(θ)
    mean(abs2, Xₙ .- X̂)
end

# =============================================================================
#  4. Train the network (ADAM warm-up -> L-BFGS polish)
# =============================================================================
losses = Float64[]

callback = function (state, l)
    push!(losses, l)
    if length(losses) % 50 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p))

# Stage 1 — ADAM: fast, robust warm-up.
res1 = Optimization.solve(
    optprob, OptimizationOptimisers.Adam(1e-3), callback = callback, maxiters = 5000)
println("Training loss after $(length(losses)) iterations: $(losses[end])")

# Stage 2 — L-BFGS: high-accuracy polish from the ADAM solution.
optprob2 = Optimization.OptimizationProblem(optf, res1.u)
res2 = Optimization.solve(
    optprob2, LBFGS(linesearch = BackTracking()), callback = callback, maxiters = 5000)
println("Final training loss after $(length(losses)) iterations: $(losses[end])")

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

# Training-loss curve (ADAM vs L-BFGS).
pl_losses = plot(1:5000, losses[1:5000], yaxis = :log10, xaxis = :log10,
    xlabel = "Iterations", ylabel = "Loss", label = "ADAM", color = :blue)
plot!(5001:length(losses), losses[5001:end], yaxis = :log10, xaxis = :log10,
    xlabel = "Iterations", ylabel = "Loss", label = "LBFGS", color = :red)
savefig(pl_losses, joinpath(FIG_DIR, "02_training_losses.png"))

# =============================================================================
#  5. Analyse the trained model
# =============================================================================
# Trained trajectory on a finer time grid vs. the ground truth.
ts = first(solution.t):(mean(diff(solution.t)) / 2):last(solution.t)
X̂ = predict(p_trained, Xₙ[:, 1], ts)
pl_trajectory = plot(ts, transpose(X̂), xlabel = "t", ylabel = "x(t), y(t)", color = :red,
    label = ["UDE Approximation" nothing])
scatter!(solution.t, transpose(Xₙ), color = :black, label = ["Ground truth" nothing])
savefig(pl_trajectory, joinpath(FIG_DIR, "03_ude_trajectory.png"))

# What the network learned vs. the true missing interaction term.
Ȳ = [-p_[2] * (X̂[1, :] .* X̂[2, :])'; p_[3] * (X̂[1, :] .* X̂[2, :])']   # true interaction
Ŷ = U(X̂, p_trained, st)[1]                                             # network guess
pl_reconstruction = plot(ts, transpose(Ŷ), xlabel = "t", ylabel = "U(x,y)", color = :red,
    label = ["UDE Approximation" nothing])
plot!(ts, transpose(Ȳ), color = :black, label = ["True Interaction" nothing])
savefig(pl_reconstruction, joinpath(FIG_DIR, "04_missing_term_reconstruction.png"))

# Reconstruction error over time, plus combined overview panels.
pl_reconstruction_error = plot(ts, norm.(eachcol(Ȳ - Ŷ)), yaxis = :log, xlabel = "t",
    ylabel = "L2-Error", label = nothing, color = :red)
pl_missing = plot(pl_reconstruction, pl_reconstruction_error, layout = (2, 1))
pl_overall = plot(pl_trajectory, pl_missing)
savefig(pl_missing, joinpath(FIG_DIR, "05_reconstruction_and_error.png"))
savefig(pl_overall, joinpath(FIG_DIR, "06_overall.png"))

# =============================================================================
#  6. Sparse symbolic regression — recover the interaction as an equation
# =============================================================================
# Polynomial candidate library, both as callable functions (to build the design
# matrix) and as symbolic expressions (to pretty-print the result). `@variables`
# is provided by ModelingToolkit, loaded above.
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

# Design matrix Φ (rows = samples, cols = basis terms) and the two regression
# targets (the components of the network output along the trajectory).
X̂ = Array(X̂)
Ŷ = Array(Ŷ)
Nt = size(X̂, 2)
inputs = [X̂[:, i] for i in 1:Nt]
targets = [Ŷ[:, i] for i in 1:Nt]
Φ = [Float64(f(u)) for u in inputs, f in basis_funcs]
y₁ = reshape(Float64.([y[1] for y in targets]), :, 1)
y₂ = reshape(Float64.([y[2] for y in targets]), :, 1)

# LASSO ( min ‖Φβ − y‖₂² + λ‖β‖₁ ) via coordinate descent with soft-thresholding.
# The L1 penalty drives all but the genuinely-active basis terms to zero, so the
# regression SELECTS a sparse, interpretable set of terms.
λ = 0.1          # LASSO strength (controls selection)
threshold = 0.1  # a basis term counts as "active" if |coefficient| exceeds this

function lasso_cd(Φ, y; λ = 0.1, iters = 50_000, tol = 1e-12)
    m = size(Φ, 2)
    β = zeros(m)
    colsq = vec(sum(Φ .^ 2, dims = 1))
    soft(z, γ) = sign(z) * max(abs(z) - γ, 0.0)
    r = y .- Φ * β                                   # maintained residual y − Φβ
    for _ in 1:iters
        Δ = 0.0
        for j in 1:m
            colsq[j] == 0 && continue
            ρ = dot(view(Φ, :, j), r) + colsq[j] * β[j]   # Φⱼ ⋅ (y − Φβ + Φⱼβⱼ)
            βⱼ = soft(ρ, λ / 2) / colsq[j]
            d = βⱼ - β[j]
            if d != 0
                r .-= d .* view(Φ, :, j)                  # rank-1 residual update
                β[j] = βⱼ
                Δ = max(Δ, abs(d))
            end
        end
        Δ < tol && break
    end
    return reshape(β, :, 1)
end

# Relaxed LASSO: use the L1 fit only to SELECT the active terms, then refit their
# magnitudes by ordinary least squares. LASSO is excellent at selection but biases
# coefficients toward zero, so this debiasing step recovers their true size.
function debias(β, Φ, y; threshold = 0.1)
    active = findall(c -> abs(c) > threshold, vec(β))
    βr = zeros(size(Φ, 2))
    isempty(active) || (βr[active] = Φ[:, active] \ vec(y))
    return reshape(βr, :, 1)
end

β₁ = debias(lasso_cd(Φ, vec(y₁); λ = λ), Φ, y₁; threshold = threshold)
β₂ = debias(lasso_cd(Φ, vec(y₂); λ = λ), Φ, y₂; threshold = threshold)

# Pretty-print the surviving terms.
function format_expr(β, φs, threshold)
    terms = String[]
    for (coeff, term) in zip(β, φs)
        if abs(coeff) > threshold
            push!(terms, @sprintf("%f", coeff) * "*" * string(term))
        end
    end
    return join(terms, " + ")
end

expr₁_str = format_expr(β₁, basis_symbols, threshold)
expr₂_str = format_expr(β₂, basis_symbols, threshold)

println("\nLearned symbolic expression for NN output 1 (thresholded):")
println("  y₁(t) ≈ ", expr₁_str)
println("Learned symbolic expression for NN output 2 (thresholded):")
println("  y₂(t) ≈ ", expr₂_str)

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
#  7. Rebuild the mechanistic model from the recovered coefficients
# =============================================================================
# The regression keeps only the u₁·u₂ term (index 7); its coefficient is the
# interaction strength. Since the true terms are Ȳ₁ = -β·xy and Ȳ₂ = +γ·xy,
# we read β = -β₁[7] and γ = β₂[7] straight from the recovered coefficients.
β_recovered = -β₁[7]
γ_recovered = β₂[7]
println("\nRecovered interaction coefficients:")
println("  β (prey loss)     ≈ ", β_recovered, "   (true 0.9)")
println("  γ (predator gain) ≈ ", γ_recovered, "   (true 0.8)")

u0 = [3.1461493970111687, 1.5370475785612603]
params_actual = [1.3, 0.9, 0.8, 1.8]
params_learned = [1.3, β_recovered, γ_recovered, 1.8]

sol_actual = solve(ODEProblem(lotka!, u0, tspan, params_actual),
    Tsit5(), abstol = 1e-12, reltol = 1e-12, saveat = 0.025)
sol_learned = solve(ODEProblem(lotka!, u0, tspan, params_learned),
    Tsit5(), abstol = 1e-12, reltol = 1e-12, saveat = 0.025)

fig_compare = plot(sol_actual.t, sol_actual[1, :], label = "Prey - Actual", lw = 2)
plot!(sol_actual.t, sol_actual[2, :], label = "Predator - Actual", lw = 2)
plot!(sol_learned.t, sol_learned[1, :], label = "Prey - Learned", ls = :dash, lw = 2)
plot!(sol_learned.t, sol_learned[2, :], label = "Predator - Learned", ls = :dash, lw = 2)
xlabel!("Time")
ylabel!("Population")
title!("Actual vs Recovered Lotka–Volterra Model")
savefig(fig_compare, joinpath(FIG_DIR, "07_actual_vs_learned.png"))

println("\nDone. Figures saved to $(FIG_DIR)")
println("Numeric artefacts saved to $(RES_DIR)")
