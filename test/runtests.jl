using ObjConsNLPModels
using Test
using Percival
using NLPModels
# using NLPModelsTest

@testset "square" begin
    lvar = [0.0, 0.0]
    uvar = [2.0, 2.0]
    model = objcons_nlpmodel(; x0 = [1.99, 1.99], lvar, uvar) do x
        @assert all(lvar .≤ x .≤ uvar)
        [sum(abs2, x), sum(x) - 1]
    end
    @test get_nvar(model) == 2
    @test get_ncon(model) == 1
    @test get_lvar(model) == lvar
    @test get_uvar(model) == uvar
    x = rand(2)
    @test obj(model, x) == sum(abs2, x)
    @test cons(model, x) == [sum(x) - 1]
    @test grad(model, x) == 2 .* x
    v = rand(2)
    @test jprod(model, x, v) == [sum(v)]

    output = percival(model)
    @test output.status == :first_order
    @test output.solution ≈ [0.5, 0.5]
    @test output.objective ≈ 0.5 atol = 1e-4
end

@testset "non-square" begin
    model = objcons_nlpmodel(x -> [sum(abs2, x), sum(x) - 1]; x0 = [2.0, 2.0, 2.0])
    @test get_nvar(model) == 3
    @test get_ncon(model) == 1
    x = rand(3)
    @test obj(model, x) == sum(abs2, x)
    @test cons(model, x) == [sum(x) - 1]
    @test grad(model, x) == 2 .* x
    v = rand(3)
    @test jprod(model, x, v) == [sum(v)]

    output = percival(model)
    @test output.status == :first_order
    @test output.solution ≈ ones(3) ./ 3
    @test output.objective ≈ 1 / 3 atol = 1e-5

    # just do a lot of evaluations to trigger cache cleaning
    for _ in 1:10000
        @test isfinite(obj(model, randn(3)))
    end
end

# NOTE: checks below are WIP, just implemented what is needed for Percival
# NLPModelsTest.check_nlp_dimensions(model)
