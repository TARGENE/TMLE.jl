single_dimensional_value(Ψ̂) = only(estimate(Ψ̂))

HypothesisTests.OneSampleTTest(Ψ̂, Ψ₀=0) = OneSampleTTest(single_dimensional_value(Ψ̂), std(Ψ̂), Ψ̂.n, Ψ₀)

HypothesisTests.OneSampleZTest(Ψ̂, Ψ₀=0) = OneSampleZTest(single_dimensional_value(Ψ̂), std(Ψ̂), Ψ̂.n, Ψ₀)

function HypothesisTests.OneSampleHotellingT2Test(Ψ̂, Ψ₀=zeros(size(Ψ̂.estimates, 1)))
    x̄ = estimate(Ψ̂)
    S = Ψ̂.cov
    n, p = Ψ̂.n, length(x̄)
    p == length(Ψ₀) ||
        throw(DimensionMismatch("Number of variables does not match number of means"))
    n > 0 || throw(ArgumentError("The input must be non-empty"))
    
    T² = n * HypothesisTests.At_Binv_A(x̄ .- Ψ₀, S)
    F = (n - p) * T² / (p * (n - 1))
    return OneSampleHotellingT2Test(T², F, n, p, Ψ₀, x̄, S)
end

"""
    significance_test(estimate::EICEstimate, Ψ₀=0)

Performs a TTest
"""
significance_test(estimate::EICEstimate, Ψ₀=0) = OneSampleTTest(estimate, Ψ₀)

"""
    significance_test(estimate::Union{JointEstimate, ComposedEstimate}, Ψ₀=zeros(size(estimate.estimate, 1)))

Performs a TTest if the estimate is one dimensional and a HotellingT2Test otherwise.
"""
function significance_test(Ψ̂::Union{JointEstimate, ComposedEstimate}, Ψ₀=zeros(size(Ψ̂.estimates, 1)))
    if length(Ψ̂.estimates) == 1
        return OneSampleTTest(Ψ̂, only(Ψ₀))
    else
        return OneSampleHotellingT2Test(Ψ̂, Ψ₀)
    end
end