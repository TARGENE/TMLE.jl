"""
Q and G are two logistic models
"""
function binary_outcome_binary_treatment_pb(;n=100)
    rng = StableRNG(123)
    p_w() = 0.3
    pa_given_w(w) = 1 ./ (1 .+ exp.(-0.5w .+ 1))
    py_given_aw(a, w) = 1 ./ (1 .+ exp.(2w .- 3a .+ 1))
    # Sample from dataset
    Unif = Uniform(0, 1)
    W = rand(rng, Unif, n) .< p_w()
    T = rand(rng, Unif, n) .< pa_given_w(W)
    Y = rand(rng, Unif, n) .< py_given_aw(T, W)
    dataset = DataFrame(T=categorical(T), W=W, Y=categorical(Y))
    
    # Compute the theoretical ATE
    ATE₁ = py_given_aw(1, 1)*p_w() + (1-p_w())*py_given_aw(1, 0)
    ATE₀ = py_given_aw(0, 1)*p_w() + (1-p_w())*py_given_aw(0, 0)
    ATE = ATE₁ - ATE₀

    return dataset, ATE
end

"""
From https://www.degruyter.com/document/doi/10.2202/1557-4679.1043/html
"""
function continuous_outcome_binary_treatment_pb(;n=100, rng = StableRNG(123))
    # Dataset
    Unif = Uniform(0, 1)
    W = float(rand(rng, Bernoulli(0.5), n, 3))
    W₁, W₂, W₃ = W[:, 1], W[:, 2], W[:, 3]
    t = rand(rng, Unif, n) .< logistic.(0.5W₁ + 1.5W₂ - W₃)
    y = 4t + 25W₁ + 3W₂ - 4W₃ + rand(rng, Normal(0, 0.1), n)
    T = categorical(t)
    dataset = DataFrame(T = T, W₁ = W₁, W₂ = W₂, W₃ = W₃, Y = y)
    # Theroretical ATE
    ATE = 4
    return dataset, ATE
end

function continuous_outcome_categorical_treatment_pb(;n=100, control="TT", case="AA")
    # Define dataset
    rng = StableRNG(123)
    ft(T) = (T .== "AA") - (T .== "AT") + 2(T .== "TT")
    fw(W₁, W₂, W₃) = 2W₁ + 3W₂ - 4W₃
    W = float(rand(rng, Bernoulli(0.5), n, 3))
    W₁, W₂, W₃ = W[:, 1], W[:, 2], W[:, 3]
    θ = rand(rng, 3, 3)
    softmax = exp.(W*θ) ./ sum(exp.(W*θ), dims=2)
    T = [sample(rng, ["TT", "AA", "AT"], Weights(softmax[i, :])) for i in 1:n]
    y = ft(T) + fw(W₁, W₂, W₃) + rand(rng, Normal(0,1), n)
    dataset = DataFrame(T = categorical(T),  W₁ = W₁, W₂ = W₂, W₃ = W₃, Y = y)
    # True ATE: Ew[E[Y|t,w]] = ∑ᵤ (ft(T) + fw(w))p(w) = ft(t) + 0.5
    ATE = (ft(case) + 0.5) -  (ft(control) + 0.5)
    return dataset, ATE
end


function dataset_2_treatments_pb(;rng = StableRNG(123), n=100)
    # Dataset
    μY(W₁, W₂, T₁, T₂) = 4T₁ .- 2T₂ .+ 5W₁ .- 3W₂
    W₁ = rand(rng, Normal(), n)
    W₂ = rand(rng, Normal(), n)
    μT₁ = logistic.(0.5W₁ + 1.5W₂)
    T₁ = float(rand(rng, Uniform(), n) .< μT₁)
    μT₂ = logistic.(-1.5W₁ + .5W₂ .+ 1)
    T₂ = float(rand(rng, Uniform(), n) .< μT₂)
    y = μY(W₁, W₂, T₁, T₂) .+ rand(rng, Normal(), n)
    dataset = DataFrame(
        W₁ = W₁,
        W₂ = W₂,
        T₁ = categorical(T₁),
        T₂ = categorical(T₂),
        Y  = y
    )
    # Those ATEs are MC approximations, only reliable with large samples
    case = ones(n)
    control = zeros(n)
    ATE₁₁₋₀₁ = mean(μY(W₁, W₂, case, case) .- μY(W₁, W₂, control, case))
    ATE₁₁₋₀₀ = mean(μY(W₁, W₂, case, case) .- μY(W₁, W₂, control, control))

    return dataset, (ATE₁₁₋₀₁, ATE₁₁₋₀₀)
end