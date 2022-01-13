using Test


@time begin
    @test include("jointmodels.jl")
    @test include("utils.jl")
    @test include("model.jl")
    @test include("correctness_ate.jl")
    @test include("correctness_iate.jl")
    @test include("report.jl")
end