using Test

@time begin
    @test include("scm.jl")
    @test include("non_regression_test.jl")
    @test include("utils.jl")
    @test include("gradient.jl")
    @test include("fluctuation.jl")
    @test include("double_robustness_ate.jl")
    @test include("double_robustness_iate.jl")
    @test include("3points_interactions.jl")
    @test include("estimation.jl")
    @test include("estimands.jl")
    @test include("missing_management.jl")
    @test include("composition.jl")
    @test include("treatment_transformer.jl")
end