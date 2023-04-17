using ObjConsNLPModels
using Test
using Percival
using NLPModels
using NLPModelsTest

model = objcons_nlpmodel(x -> [sum(abs2, x), sum(x) - 1]; x0 = [2.0, 2.0])
@test get_nvar(model) == 2
@test get_ncon(model) == 1
x = rand(2)
@test obj(model, x) == sum(abs2, x)
@test cons(model, x) == [sum(x) - 1]
@test grad(model, x) == 2 .* x
v = rand(2)
@test jprod(model, x, v) == [sum(v)]

# NLPModelsTest.check_nlp_dimensions(model)

# output = percival(model)
