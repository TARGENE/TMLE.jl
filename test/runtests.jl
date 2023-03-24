using Test

@time begin
    @test include("cache.jl")
    @test include("utils.jl")
    @test include("double_robustness_ate.jl")
    @test include("double_robustness_iate.jl")
    @test include("3points_interactions.jl")
    @test include("warm_restart.jl")
    @test include("parameters.jl")
    @test include("miscellaneous.jl")
    @test include("composition.jl")
    @test include("treatment_transformer.jl")
    @test include("fit_nuisance.jl")
end