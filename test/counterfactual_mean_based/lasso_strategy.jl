using Random, Distributions, LinearAlgebra, DataFrames

function sim3(n::Int=1000, p::Int=100, rho::Float64=0.9, k::Int=20, amplitude::Float64=1, amplitude2::Float64=1, k2::Int=20)
    function toeplitz_cov(p, rho)
        return toeplitz(rho .^ (0:(p-1)))
    end
    
    Sigma = toeplitz_cov(p, rho)
    
    mu = zeros(Float64, p)
    W = (rand(MvNormal(mu, Sigma), n))'  
    W = (W .- mean(W, dims=1)) ./ std(W, dims=1) 
    
    nonzero = sample(1:p, k, replace=false)
    sign = sample([-1, 1], p, replace=true)
    gamma_ = amplitude .* sign .* in(1:p, nonzero)
    
    nonzero2 = sample(1:p, k2, replace=false)
    sign2 = sample([-1, 1], p, replace=true)
    beta_ = amplitude2 .* sign2 .* in(1:p, nonzero2)
    logit_p = W * beta_
    prob_A = 1 ./ (1 .+ exp.(-logit_p))
    A = rand(Binomial(1, prob_A))
    
    Y = 2 .* A .+ W * gamma_ .+ randn(n)
    
    data = DataFrame(hcat(W, A, Y))
    names!(data, [string("W", i) for i in 1:p]..., "A", "Y")

    return data
end

