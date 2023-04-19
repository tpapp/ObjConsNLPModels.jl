"""
Implement an `AbstractNLPModel` that provides objective, constraint, and their first derivatives.

The single exported function is [`objcons_nlpmodel`](@ref).

# Motivation

Consider a problem
```math
\\min_x f(g(x)) \\text{ subject to } h(g(x)) == 0
```
where `g` is an expensive function. A typical example is structural estimation in
macroeconomics, where `g` would calculate something expensive like mass distributions, `f`
the moments and `h` the equilibrium conditions. The implementation allows the user to define
these in one step.
"""
module ObjConsNLPModels

export objcons_nlpmodel

using ArgCheck: @argcheck
import DiffResults
using DocStringExtensions: SIGNATURES, FIELDS
import ForwardDiff
using LinearAlgebra: mul!, dot
using NLPModels: NLPModels, AbstractNLPModel, Counters, NLPModelMeta, get_nvar, DimensionError, @lencheck
using SimpleUnPack: @unpack
using SparseDiffTools: JacVec, HesVec

####
#### caching evaluations
####

Base.@kwdef struct EvaluationCache{T,S}
    "cache of evaluations"
    evaluations::Dict{T,Pair{Int,S}}
    "last evaluation index, for caching"
    last_index::Base.RefValue{Int}
    "cache is compacted to this size"
    min_size::Int
    "trigger for compacting cache, see [`maybe_compact_cache`](@ref)"
    max_size::Int
end

function evaluation_cache(domain::Type{T}, codomain::Type{S};
                          min_size::Int = 50, max_size::Int = 200) where {T,S}
    @argcheck 0 ≤ min_size ≤ max_size "Cache sizes should be nonnegative and ordered."
    EvaluationCache(; evaluations = Dict{T,Pair{Int,S}}(), last_index =  Ref(0), min_size, max_size)
end

function ensure_evaluated!(f::F, cache::EvaluationCache{T,S}, x::T) where {F,T,S}
    @unpack evaluations, last_index, min_size, max_size = cache
    indexed_y = get!(evaluations, x) do
        y = f(x)
        last_index[] += 1
        if length(evaluations) > max_size # compact cache
            keep_index = last_index[] - min_size
            filter!(entry -> entry.index ≥ keep_index, evaluations)
        end
        last_index[] => convert(S, y)::S
    end
    indexed_y[2]
end

function ensure_evaluated!(f::F, cache::EvaluationCache{T}, x) where {F,T}
    ensure_evaluated!(f, cache, convert(T, x)::T)
end

####
#### model definition
####

"Type of cache entries for objective, constraint, and derivatives. Internal."
const _OBJCONS{Z} = NamedTuple{(:objcons, :objcons_jacobian), Tuple{Vector{Z}, Matrix{Z}}}

"Type of cache entries for Hessian of the objective. Internal."
const _HESSIAN{Z} = Matrix{Z}

"""
Container for the model setup. Internal, use [`objcons_nlpmodel`](@ref) to instantiate.

$(FIELDS)
"""
Base.@kwdef struct ObjConsNLPModel{T,S,Z,F} <: AbstractNLPModel{T,S}
    "meta information for NLPModels"
    meta::NLPModelMeta{T,S}
    "counters for NLPModels"
    counters::Counters
    "function that returns `[objective, constraints...]`."
    objcons_function::F
    "Cache of previous evaluations and Jacobians."
    objcons_cache::EvaluationCache{Vector{Z},_OBJCONS{Z}}
    "Cache of objective Hessian."
    hessian_cache::EvaluationCache{Vector{Z},_HESSIAN{Z}}
end

"""
$(SIGNATURES)

Create an `AbstractNLPModel` where `objcons_function` returns `[objective, constraints...]`.
"""
function objcons_nlpmodel(objcons_function; x0::AbstractVector, lvar = fill(-Inf, length(x0)), uvar = fill(Inf, length(x0)),
                          min_cache_size = 200, max_cache_size = 500)
    @argcheck all(isfinite, x0) "Initial guess should be finite."
    nvar = length(x0)
    ncon = length(objcons_function(x0)) - 1
    @argcheck nvar == length(lvar) DimensionMismatch
    meta = NLPModelMeta(nvar; lvar, uvar, ncon, x0, lcon = zeros(ncon), ucon = zeros(ncon))
    counters = Counters()
    S = eltype(x0)
    Z = S ≡ Any ? Float64 : float(S)
    objcons_cache = evaluation_cache(Vector{Z}, _OBJCONS{Z}; min_size = min_cache_size, max_size = max_cache_size)
    hessian_cache = evaluation_cache(Vector{Z}, _HESSIAN{Z}; min_size = min_cache_size, max_size = max_cache_size)
    ObjConsNLPModel(; meta, counters, objcons_function, objcons_cache, hessian_cache)
end

"""
$(SIGNATURES)

Evaluate `objcons_function` at a point `x`, with derivatives, from the evaluations cache or
calculating as necessary. Internal.
"""
function objcons_at_point(model::ObjConsNLPModel, x::Vector)
    @unpack meta, objcons_function, objcons_cache = model
    @argcheck length(x) == get_nvar(meta) DimensionError("x", get_nvar(meta), length(x))
    ensure_evaluated!(objcons_cache, x) do x
        AD_result = DiffResults.JacobianResult(x)
        ForwardDiff.jacobian!(AD_result, objcons_function, x)
        objcons = DiffResults.value(AD_result)
        objcons_jacobian = DiffResults.jacobian(AD_result)
        (; objcons, objcons_jacobian)
    end
end

"""
$(SIGNATURES)

Return the hessian of the objective at a point `x`, with derivatives, from the evaluations cache or
calculating as necessary. Internal.
"""
function hessian_at_point(model::ObjConsNLPModel, x::Vector)
    @argcheck length(x) == get_nvar(model.meta) DimensionError("x", get_nvar(model), length(x))
    ensure_evaluated!(model.hessian_cache, x) do x
        ForwardDiff.hessian(x -> first(objcons(x)), x)
    end
end

####
#### supported API
####

function NLPModels.obj(model::ObjConsNLPModel, x::AbstractVector)
    objcons_at_point(model, x).objcons[1]
end

function NLPModels.cons(model::ObjConsNLPModel, x::AbstractVector)
    objcons_at_point(model, x).objcons[(begin+1):end]
end

function NLPModels.cons!(model::ObjConsNLPModel, x::AbstractVector, buffer::AbstractVector)
    copy!(buffer, @view objcons_at_point(model, x).objcons[(begin+1):end])
end

function NLPModels.objcons(model::ObjConsNLPModel, x::AbstractVector)
    objcons = objcons_at_point(model, x).objcons
    obj = first(objcons)
    cons = objcons[(begin+1):end]
    obj, cons
end

function NLPModels.objcons!(model::ObjConsNLPModel, x::AbstractVector, buffer::AbstractVector)
    objcons = objcons_at_point(model, x).objcons
    copy!(buffer, @view objcons[(begin+1):end])
    first(objcons), buffer
end

function NLPModels.grad(model::ObjConsNLPModel, x::AbstractVector)
    objcons_at_point(model, x).objcons_jacobian[1, :]
end

function NLPModels.grad!(model::ObjConsNLPModel, x::AbstractVector, buffer::AbstractVector)
    @lencheck get_nvar(model) buffer
    copy!(buffer, @view objcons_at_point(model, x).objcons_jacobian[1, :])
end

function _jacobian(model, x)
    @view(objcons_at_point(model, x).objcons_jacobian[(begin + 1):end, :])
end

function NLPModels.jprod(model::ObjConsNLPModel, x::AbstractVector, v::AbstractVector)
    _jacobian(model, x) * v
end

function NLPModels.jprod!(model::ObjConsNLPModel, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
    mul!(Jv, _jacobian(model, x), v)
end

function NLPModels.jtprod(model::ObjConsNLPModel, x::AbstractVector, v::AbstractVector)
    _jacobian(model, x)' * v
end

function NLPModels.jtprod!(model::ObjConsNLPModel, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
    mul!(Jv, _jacobian(model, x)', v)
end

function NLPModels.hprod!(model::ObjConsNLPModel, x::AbstractVector, v::AbstractVector, Hv::AbstractVector; obj_weight = 1.0)
    @lencheck get_nvar(model) x v Hv
    mul!(Hv, hessian_at_point(model, x), v, obj_weight, 0.0)
end

function NLPModels.hprod!(model::ObjConsNLPModel, x::AbstractVector, y::AbstractVector, v::AbstractVector, Hv::AbstractVector; obj_weight = 1.0)
    @lencheck get_nvar(model) x y v Hv
    @unpack objcons_function = model
    function f(x)
        oc = objcons_function(x)
        obj_weight * oc[begin] + dot(y, @view oc[(begin+1):end])
    end
    mul!(Hv, HesVec(f, x), v)
end

end # module
