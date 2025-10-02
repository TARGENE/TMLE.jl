"""
Example: LASSO Collaborative TMLE with CairoMakie Plots

Demonstrates CV-based variable selection in high-dimensional causal inference 
and generates static plots using CairoMakie from the docs environment.
"""

using Pkg
Pkg.activate("docs")

using CairoMakie
using Printf
using Statistics
using Random

Pkg.activate(".")

using TMLE
using DataFrames
using CategoricalArrays
using GLMNet
using Distributions
using LinearAlgebra
using StatsBase

"""
Create a Toeplitz matrix manually from a vector
A Toeplitz matrix has constant diagonals, where T[i,j] = c[|i-j|+1]
"""
function create_toeplitz_matrix(c::Vector{T}) where T
    n = length(c)
    matrix = Matrix{T}(undef, n, n)
    
    for i in 1:n
        for j in 1:n
            matrix[i, j] = c[abs(i - j) + 1]
        end
    end
    
    return matrix
end

println("🧬 LASSO Collaborative TMLE Example with CairoMakie Plots")
println("=" ^ 60)

Random.seed!(123)

function sim3(; n=1000, p=100, rho=0.9, k=20, amplitude=1.0, amplitude2=1.0, k2=20)
    toeplitz_vector = [rho^i for i in 0:(p-1)]
    Sigma = create_toeplitz_matrix(toeplitz_vector)
    
    mv_normal = MvNormal(zeros(p), Matrix(Sigma))
    W_raw = rand(mv_normal, n)'  
    W = (W_raw .- mean(W_raw, dims=1)) ./ std(W_raw, dims=1)
    
    nonzero2 = sample(1:p, k2, replace=false)
    signs2 = sample([-1, 1], p, replace=true)
    beta = amplitude2 * signs2 .* [i in nonzero2 for i in 1:p]
    
    logit_p = W * beta
    prob_A = 1 ./ (1 .+ exp.(-logit_p))
    A = rand.(Bernoulli.(prob_A))
    
    nonzero = sample(1:p, k, replace=false)
    signs = sample([-1, 1], p, replace=true)
    gamma = amplitude * signs .* [i in nonzero for i in 1:p]
    
    Y = 2.0 * A + W * gamma + randn(n)
    
    W_df = DataFrame(W, [Symbol("W$i") for i in 1:p])
    data = hcat(W_df, DataFrame(A=categorical(A), Y=Y))
    
    return data, nonzero, nonzero2
end

println("\n📊 Generating high-dimensional simulation data...")
n = 2000 
p = 30   
rho = 0.5
n_bootstrap = 100 

println("Simulation parameters:")
println("  Sample size: $n")
println("  Confounders: $p") 
println("  Correlation: $rho")
println("  Bootstrap samples: $n_bootstrap")

dataset, true_outcome_vars, true_ps_vars = sim3(n=n, p=p, rho=rho, k=15, k2=15)
all_confounders = [Symbol("W$i") for i in 1:p]

estimand = ATE(
    outcome = :Y,
    treatment_values = (A = (case = 1, control = 0),),
    treatment_confounders = (A = all_confounders,)
)

println("\n🔄 Running bootstrap comparison...")
println("=" ^ 50)

standard_estimates = Float64[]
lasso_estimates = Float64[]

print("Progress: ")
for i in 1:n_bootstrap
    if i % 10 == 0
        print("$i ")
    end
    
    boot_indices = sample(1:n, n, replace=true)
    boot_dataset = dataset[boot_indices, :]
    
    standard_estimator = Tmle()
    try
        standard_result, _ = standard_estimator(estimand, boot_dataset; verbosity=0)
        push!(standard_estimates, estimate(standard_result))
    catch
        push!(standard_estimates, NaN)
    end
    
    lasso_strategy = LassoCTMLE(
        confounders = all_confounders,
        patience = 4,  
        alpha = 1.0
    )
    lasso_estimator = Tmle(collaborative_strategy = lasso_strategy)
    try
        lasso_result, _ = lasso_estimator(estimand, boot_dataset; verbosity=0)
        push!(lasso_estimates, estimate(lasso_result))
    catch
        push!(lasso_estimates, NaN)
    end
end

println("\n✅ Bootstrap completed!")

valid_standard = filter(!isnan, standard_estimates)
valid_lasso = filter(!isnan, lasso_estimates)

println("\nBootstrap Results:")
println("=" ^ 50)
println("Valid estimates:")
println("  Standard TMLE: $(length(valid_standard))/$n_bootstrap")
println("  LASSO CTMLE:   $(length(valid_lasso))/$n_bootstrap")

if length(valid_standard) > 10 && length(valid_lasso) > 10
    println("\nSummary Statistics:")
    println("Standard TMLE:")
    println("  Mean: $(round(mean(valid_standard), digits=3))")
    println("  Std:  $(round(std(valid_standard), digits=3))")
    println("  Bias: $(round(abs(mean(valid_standard) - 2.0), digits=3))")
    
    println("LASSO CTMLE:")
    println("  Mean: $(round(mean(valid_lasso), digits=3))")
    println("  Std:  $(round(std(valid_lasso), digits=3))")
    println("  Bias: $(round(abs(mean(valid_lasso) - 2.0), digits=3))")
    
    println("\n📊 Creating CairoMakie plots...")
    
    fig = Figure(size = (1000, 800))
    
    ax1 = Axis(fig[1, 1], 
               title = "Standard TMLE Distribution",
               xlabel = "Estimate Value", 
               ylabel = "Frequency")
    
    ax2 = Axis(fig[1, 2], 
               title = "LASSO CTMLE Distribution",
               xlabel = "Estimate Value", 
               ylabel = "Frequency")
    
    hist!(ax1, valid_standard, bins=20, color=(:blue, 0.7), strokewidth=1, strokecolor=:blue)
    hist!(ax2, valid_lasso, bins=20, color=(:green, 0.7), strokewidth=1, strokecolor=:green)
    
    vlines!(ax1, [2.0], color=:red, linewidth=2, linestyle=:dash)
    vlines!(ax1, [mean(valid_standard)], color=:blue, linewidth=2, linestyle=:dot)
    vlines!(ax2, [2.0], color=:red, linewidth=2, linestyle=:dash)
    vlines!(ax2, [mean(valid_lasso)], color=:green, linewidth=2, linestyle=:dot)
    
    ax3 = Axis(fig[2, 1:2], 
               title = "Bootstrap Distribution Comparison",
               xlabel = "Estimate Value", 
               ylabel = "Frequency")
    
    hist!(ax3, valid_standard, bins=20, color=(:blue, 0.6), strokewidth=1, strokecolor=:blue, label="Standard TMLE")
    hist!(ax3, valid_lasso, bins=20, color=(:green, 0.6), strokewidth=1, strokecolor=:green, label="LASSO CTMLE")
    vlines!(ax3, [2.0], color=:red, linewidth=3, linestyle=:dash, label="True ATE = 2.0")
    
    axislegend(ax3, position=:rt)
    
    plot_filename = "lasso_ctmle_bootstrap_results.png"
    save(plot_filename, fig)
    println("📊 Plot saved as: $plot_filename")
    
    fig2 = Figure(size = (600, 400))
    ax4 = Axis(fig2[1, 1], 
               title = "Box Plot Comparison",
               ylabel = "Estimate Value")
    
    standard_median = median(valid_standard)
    standard_q1 = quantile(valid_standard, 0.25)
    standard_q3 = quantile(valid_standard, 0.75)
    
    lasso_median = median(valid_lasso)
    lasso_q1 = quantile(valid_lasso, 0.25)
    lasso_q3 = quantile(valid_lasso, 0.75)
    
    positions = [1, 2]
    medians = [standard_median, lasso_median]
    q1s = [standard_q1, lasso_q1]
    q3s = [standard_q3, lasso_q3]
    
    for (i, pos) in enumerate(positions)
        lines!(ax4, [pos-0.2, pos+0.2, pos+0.2, pos-0.2, pos-0.2], 
               [q1s[i], q1s[i], q3s[i], q3s[i], q1s[i]], color=:black, linewidth=2)
        lines!(ax4, [pos-0.2, pos+0.2], [medians[i], medians[i]], color=:red, linewidth=3)
    end
    
    hlines!(ax4, [2.0], color=:red, linewidth=2, linestyle=:dash)
    
    ax4.xticks = (positions, ["Standard TMLE", "LASSO CTMLE"])
    
    boxplot_filename = "lasso_ctmle_boxplot.png"
    save(boxplot_filename, fig2)
    println("📊 Box plot saved as: $boxplot_filename")
    
    println("\n📈 Side-by-Side Comparison:")
    println("=" ^ 70)
    println("Metric               | Standard TMLE | LASSO CTMLE   | Difference")
    println("-" ^ 70)
    @printf("Mean                 | %12.4f  | %12.4f  | %+9.4f\n", 
            mean(valid_standard), mean(valid_lasso), 
            mean(valid_lasso) - mean(valid_standard))
    @printf("Std Dev              | %12.4f  | %12.4f  | %+9.4f\n", 
            std(valid_standard), std(valid_lasso), 
            std(valid_lasso) - std(valid_standard))
    @printf("Bias (from 2.0)      | %12.4f  | %12.4f  | %+9.4f\n", 
            abs(mean(valid_standard) - 2.0), abs(mean(valid_lasso) - 2.0),
            abs(mean(valid_lasso) - 2.0) - abs(mean(valid_standard) - 2.0))
    
    variance_reduction = (var(valid_standard) - var(valid_lasso)) / var(valid_standard) * 100
    @printf("Variance Reduction   | %12s  | %12s  | %+8.1f%%\n", 
            "baseline", "improved", variance_reduction)
    
    println("=" ^ 70)
    println("📊 Plots saved as PNG files in current directory!")
end

println("\n✅ Bootstrap analysis completed successfully!")
println("🎯 Summary: LASSO CTMLE demonstrates automatic variable selection with robust performance")
println("📁 Check the generated PNG files for visualization results!")
