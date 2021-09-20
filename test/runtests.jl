using Test


@time begin
    @test include("fluctuations.jl")
    @test include("jointmodels.jl")
    @test include("utils.jl")
    @test include("api.jl")
    @test include("double_robustness_ate.jl")
    @test include("double_robustness_iate.jl")
end