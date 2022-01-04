using Test


@time begin
    @test include("jointmodels.jl")
    @test include("utils.jl")
    @test include("model.jl")
    @test include("double_robustness_ate.jl")
    @test include("double_robustness_iate.jl")
end