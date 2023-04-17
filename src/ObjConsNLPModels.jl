"""
Implement an `AbstractNLPModel` that provides objective, constraint, and their first derivatives.

The single exported function is [`objcons_nlpmodel`](@ref).

# Motivation

Consider a problem
```math
\min_x f(g(x)) \text{ subject to } h(g(x)) == 0
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
using NLPModels: NLPModels, AbstractNLPModel, Counters, NLPModelMeta, get_nvar, DimensionError, @lencheck
using SimpleUnPack: @unpack

"Type of cache entries. Internal."
const _CACHED{Z} = NamedTuple{(:index, :objcons, :objcons_jacobian), Tuple{Int, Vector{Z}, Matrix{Z}}}

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
    "Cache of previous evaluations. Indexed by an integer that increases for each evaluation."
    cache::Dict{Vector{Z},_CACHED{Z}}
    "cache is compacted to this size"
    min_cache_size::Int
    "trigger for compacting cache, see [`maybe_compact_cache`](@ref)"
    max_cache_size::Int
    "last evaluation index, for caching"
    last_index::Base.RefValue{Int}
end

"""
$(SIGNATURES)

Create an `AbstractNLPModel` where `objcons_function` returns `[objective, constraints...]`.
"""
function objcons_nlpmodel(objcons_function; x0::AbstractVector, lvar = fill(-Inf, length(x0)), uvar = fill(Inf, length(x0)),
                          min_cache_size = 200, max_cache_size = 500)
    @argcheck 0 ≤ min_cache_size ≤ max_cache_size "Cache sizes should be nonnegative and ordered."
    @argcheck all(isfinite, x0) "Initial guess should be finite."
    nvar = length(x0)
    ncon = length(objcons_function(x0)) - 1
    @argcheck nvar == length(lvar) DimensionMismatch
    meta = NLPModelMeta(nvar; lvar, uvar, ncon, x0)
    counters = Counters()
    S = eltype(x0)
    Z = S ≡ Any ? Float64 : float(S)
    cache = Dict{Vector{Z},_CACHED{Z}}()
    last_index = Ref(0)
    ObjConsNLPModel(; meta, counters, objcons_function, cache, min_cache_size, max_cache_size, last_index)
end

"""
$(SIGNATURES)

Evaluate `objconst` at a point `x`, with derivatives, and return `(; objcons,
objcons_jacobian)`. No caching is performed. Internal.
"""
function evaluate_at_point(objcons::F, x::Vector) where F
    AD_result = DiffResults.JacobianResult(x)
    ForwardDiff.jacobian!(AD_result, objcons, x)
    objcons = DiffResults.value(AD_result)
    objcons_jacobian = DiffResults.jacobian(AD_result)
    (; objcons, objcons_jacobian)
end

"""
$(SIGNATURES)

Compact cache if necessary.
"""
function maybe_compact_cache!(model::ObjConsNLPModel)
    @unpack cache, min_cache_size, max_cache_size, last_index = model
    if length(cache) > max_cache_size
        keep_index = last_index - min_cache_size
        filter!(entry -> entry.index ≥ keep_index, cache)
    end
    nothing
end

"""
$(SIGNATURES)

When `x` is in the cache, it is looked up, otherwise an evaluation in performed.

The result has properties `objcons` and `objcons_jacobian`. Callers should not expose these
directly (to avoid accidental cache corruption), but make copies.
"""
function ensure_evaluated(model::ObjConsNLPModel{T,S,Z}, x::Vector{Z}) where {T,S,Z}
    @argcheck length(x) == get_nvar(model.meta) DimensionError("x", get_nvar(model), length(x))
    result = get!(model.cache, x) do
        r = evaluate_at_point(model.objcons_function, x)
        model.last_index[] += 1
        _CACHED{Z}((; index = model.last_index[], r...))
    end
    maybe_compact_cache!(model)
    result
end

####
#### supported API
####

function NLPModels.obj(model::ObjConsNLPModel, x::AbstractVector)
    ensure_evaluated(model, x).objcons[1]
end

function NLPModels.cons(model::ObjConsNLPModel, x::AbstractVector)
    ensure_evaluated(model, x).objcons[(begin+1):end]
end

function NLPModels.cons!(model::ObjConsNLPModel, x::AbstractVector, buffer::AbstractVector)
    copy(buffer, @view ensure_evaluated(model, x).objcons[(begin+1):end])
end

function NLPModels.objcons(model::ObjConsNLPModel, x::AbstractVector)
    copy(ensure_evaluated(model, x).objcons)
end

function NLPModels.objcons!(model::ObjConsNLPModel, x::AbstractVector, buffer::AbstractVector)
    copy!(c, ensure_evaluated(model, x).objcons)
end

function NLPModels.grad(model::ObjConsNLPModel, x::AbstractVector)
    ensure_evaluated(model, x).objcons_jacobian[1, :]
end

function NLPModels.grad!(model::ObjConsNLPModel, x::AbstractVector, buffer::AbstractVector)
    copy!(buffer, @view ensure_evaluated(model, x).objcons_jacobian[1, :])
end

function _jacobian(model, x)
    @view(ensure_evaluated(model, x).objcons_jacobian[(begin + 1):end, :])
end

function NLPModels.jprod(model::ObjConsNLPModel, x::AbstractVector, v::AbstractVector)
    _jacobian(model, x) * v
end

function NLPModels.jtprod(model::ObjConsNLPModel, x::AbstractVector, v::AbstractVector)
    _jacobian(model, x)' * v
end

end # module
