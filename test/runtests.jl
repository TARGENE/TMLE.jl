using Test


@time begin
    @test include("jointmodels.jl")
    @test include("utils.jl")
    @test include("double_robustness_ate.jl")
    @test include("double_robustness_iate.jl")
    @test include("warm_restart.jl")
    @test include("parameters.jl")
end