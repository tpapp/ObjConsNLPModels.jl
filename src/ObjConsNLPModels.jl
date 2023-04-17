"""

See [`objcons_nlpmodel`](@ref).
"""
module ObjConsNLPModels

export objcons_nlpmodel

using ArgCheck: @argcheck
import DiffResults
using DocStringExtensions: SIGNATURES, FIELDS
import ForwardDiff
using NLPModels: NLPModels, AbstractNLPModel, Counters, NLPModelMeta, get_nvar, DimensionError, @lencheck
using SimpleUnPack: @unpack

const _CACHED{Z} = NamedTuple{(:index, :objcons, :objcons_jacobian), Tuple{Int, Vector{Z}, Matrix{Z}}}

Base.@kwdef struct ObjConsNLPModel{T,S,Z,F} <: AbstractNLPModel{T,S}
    meta::NLPModelMeta{T,S}
    counters::Counters
    objcons_function::F
    cache::Dict{Vector{Z},_CACHED{Z}}
    min_cache_size::Int
    max_cache_size::Int
    last_index::Base.RefValue{Int}
end

"""
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

function evaluate_at_point(objcons::F, x::Vector) where F
    AD_result = DiffResults.JacobianResult(x)
    ForwardDiff.jacobian!(AD_result, objcons, x)
    objcons = DiffResults.value(AD_result)
    objcons_jacobian = DiffResults.jacobian(AD_result)
    (; objcons, objcons_jacobian)
end

function maybe_compact_cache!(model::ObjConsNLPModel)
    @unpack cache, min_cache_size, max_cache_size, last_index = model
    if length(cache) > max_cache_size
        keep_index = last_index - min_cache_size
        filter!(entry -> entry.index ≥ keep_index, cache)
    end
    nothing
end

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
