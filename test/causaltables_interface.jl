module TestCausalTablesInterface

using TMLE
using CausalTables
using Test
using Distributions

@testset "Fit TMLE and OSE on CausalTable" begin
    scm = StructuralCausalModel(@dgp(
        W ~ Beta(2,2),
        A ~ Binomial.(1, W),
        Y ~ (@. Normal(A + W, 0.5))
    ); treatment = :A, response = :Y)

    ct = rand(scm, 100)
    Ψ = ATE(outcome = :Y, treatment_values = (A = (case = 1, control = 0),))

    # Test Tmle
    estimator = Tmle()
    result, cache = estimator(Ψ, ct, verbosity=0)
    @test result isa TMLE.TMLEstimate

    # Test OSE
    estimator = Ose()
    result, _ = estimator(Ψ, ct, cache=cache, verbosity=0)
    @test result isa TMLE.OSEstimate
end

end

true
