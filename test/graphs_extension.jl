module TestGraphsTMLEExt

using Test
using TMLE
using Graphs
using CairoMakie
using GraphMakie

@testset "Test DAG and graphplot" begin
    scm = SCM(
        SE(:Y, [:T₁, :T₂, :W₁₁, :W₁₂, :W₂₁, :W₂₂, :C]),
        SE(:T₁, [:W₁₁, :W₁₂]),
        SE(:T₂, [:W₂₁, :W₂₂]),
    )
    dag, nodes_mapping = TMLE.DAG(scm)
end

end

true