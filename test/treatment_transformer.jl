module TestTreatmentTransformer

using Test
using TMLE
using CategoricalArrays
using MLJBase

@testset "Test TreatmentTransformer" begin
    # A will be untouched
    # B will be one-hot encoded
    # C will be converted to float representation
    X = (
        A = [1, 2, 3, 4, 5],
        B = categorical([0, 1, 2, 1, 3]),
        C = categorical([0, 1, 2, 1, 2], ordered=true, levels=[0, 1, 2]),
    )    

    model = TMLE.TreatmentTransformer()

    mach = machine(model, X)
    fit!(mach, verbosity=0)

    Xt = transform(mach)
    @test Xt == (
        A = [1, 2, 3, 4, 5],
        B__0 = [1.0, 0.0, 0.0, 0.0, 0.0],
        B__1 = [0.0, 1.0, 0.0, 1.0, 0.0],
        B__2 = [0.0, 0.0, 1.0, 0.0, 0.0],
        C = [0.0, 1.0, 2.0, 1.0, 2.0]
    )

    Xwrong = (
        D = categorical(["A", "B", "A", "B", "C"], ordered=true),
    )
    mach = machine(model, Xwrong)
    fit!(mach, verbosity=0)

    @test_throws ArgumentError("Incompatible categorical value's underlying type for column D, please convert to `<:Real`") transform(mach)

end

end

true