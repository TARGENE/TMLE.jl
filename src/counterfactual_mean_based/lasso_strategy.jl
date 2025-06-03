#####################################################################
###                      Lasso                           ###
#####################################################################


mutable struct LassoCTMLE <: CollaborativeStrategy
    lambda_sequence :: Vector{Float64}
    lambda_index :: Int 
    g_models :: Vector{Any}
    losses :: Vector{Float64}
    best_index :: Int 
    Q_init                             
    Q0W::Vector{Float64}               
    Q1W::Vector{Float64}             
    Q::Vector{Float64}                 
    cv_ps_model                       
    targeting_results::Vector{Dict}
end


function initialise!(strategy::LassoCTMLE, Ψ)

    data = Ψ.dataset
    Y = data.Y
    A = data.A
    W = Matrix(data[:, Not([:Y, :A])])  # all covariates

    a, b = minimum(Y), maximum(Y)
    Y_scaled = (Y .- a) ./ (b - a)

    Q_init = fit(LassoPath, hcat(A, W), Y_scaled, Normal(), IdentityLink())

    Q0W = predict(Q_init, hcat(zeros(length(A)), W))
    Q1W = predict(Q_init, hcat(ones(length(A)), W))
    Q = predict(Q_init, hcat(A, W))

    ps_cv = fit(LassoPath, W, A, Binomial(), LogitLink())
    lambda_seq = ps_cv.lambda

    strategy.lambda_sequence = lambda_seq
    strategy.lambda_index = 1
    strategy.g_models = Vector{Any}(undef, length(lambda_seq))
    strategy.losses = fill(Inf, length(lambda_seq))
    strategy.best_index = 0
    strategy.Q_init = Q_init
    strategy.Q0W = Q0W
    strategy.Q1W = Q1W
    strategy.Q = Q
    strategy.a = a
    strategy.b = b
    strategy.targeting_results = Vector{Dict}(undef, length(lambda_seq))
end


function update!(strategy::LassoCTMLE, last_candidate, dataset)
    idx = strategy.lambda_index
    lambda = strategy.lambda_sequence[idx]
    W = Matrix(dataset[:, Not([:Y, :A])])
    A = dataset.A
    Y = dataset.Y

    g_model = fit(LassoModel, W, A, Binomial(), LogitLink(); lambda=lambda)
    strategy.g_models[idx] = g_model

    g1W = predict(g_model, W)
    g1W = clamp.(g1W, 1e-6, 1-1e-6)  # no division by zero

    H1 = A ./ g1W
    H0 = (1 .- A) ./ (1 .- g1W)

    offset1 = logit.(strategy.Q)
    fit1 = glm(@formula(Y ~ 0 + H1), DataFrame(Y=Y, H1=H1), Binomial(), Offset(offset1), wts=H1)
    epsilon_1 = coef(fit1)[1]

    offset0 = logit.(strategy.Q)
    fit0 = glm(@formula(Y ~ 0 + H0), DataFrame(Y=Y, H0=H0), Binomial(), Offset(offset0), wts=H0)
    epsilon_0 = coef(fit0)[1]

    Q_star1 = invlogit.(logit.(strategy.Q1W) .+ epsilon_1)
    Q_star0 = invlogit.(logit.(strategy.Q0W) .+ epsilon_0)

    predicted_Y = A .* Q_star1 .+ (1 .- A) .* Q_star0
    loss = -2 * mean(Y .* log.(predicted_Y) .+ (1 .- Y) .* log.(1 .- predicted_Y))

    strategy.losses[idx] = loss
    strategy.targeting_results[idx] = Dict(
        :Q_star1 => Q_star1,
        :Q_star0 => Q_star0,
        :epsilon_1 => epsilon_1,
        :epsilon_0 => epsilon_0
    )

    strategy.lambda_index += 1
end


function propensity_score(Ψ, strategy::LassoCTMLE)
    idx = strategy.lambda_index - 1  # last updated
    g_model = strategy.g_models[idx]
    W = Matrix(Ψ.dataset[:, Not([:Y, :A])])
    return predict(g_model, W)
end


function exhausted(strategy::LassoCTMLE)
    return strategy.lambda_index > length(strategy.lambda_sequence)
end


function finalise!(strategy::LassoCTMLE)
    best_idx = argmin(strategy.losses)
    strategy.best_index = best_idx
    best_targeting = strategy.targeting_results[best_idx]
    Q_star1 = best_targeting[:Q_star1]
    Q_star0 = best_targeting[:Q_star0]

    a = strategy.a
    b = strategy.b
    ATE = mean(Q_star1 .- Q_star0) * (b - a)
    data = strategy.dataset
    A = data.A
    Y = data.Y

    g_model = strategy.g_models[best_idx]
    W = Matrix(select(data, Not([:Y, :A])))
    g1W = predict(g_model, W)
    g1W = clamp.(g1W, 1e-6, 1 - 1e-6)
    IF = (A ./ g1W) .* (Y .- Q_star1) .- ((1 .- A) ./ (1 .- g1W)) .* (Y .- Q_star0) .+ (Q_star1 .- Q_star0) .- (ATE / (b - a))
    SE = sqrt(var(IF) / length(Y)) * (b - a)
    z = quantile(Normal(), 1 - 0.05 / 2)
    CI_lower = ATE - z * SE
    CI_upper = ATE + z * SE
    return (
        ATE = ATE,
        SE = SE,
        CI_lower = CI_lower,
        CI_upper = CI_upper,
        best_lambda = strategy.lambda_sequence[best_idx],
        loss = strategy.losses[best_idx]
    )
end

