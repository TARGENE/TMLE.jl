struct Configuration
    estimands::AbstractVector{<:Estimand}
    scm::Union{Nothing, SCM}
    adjustment::Union{Nothing, <:AdjustmentMethod}
end

Configuration(;estimands, scm=nothing, adjustment=nothing) = Configuration(estimands, scm, adjustment)

function to_dict(configuration::Configuration)
    config_dict = Dict(
        :type => "Configuration",
        :estimands => [to_dict(Ψ) for Ψ in configuration.estimands],
    )
    if configuration.scm !== nothing
        config_dict[:scm] = to_dict(configuration.scm)
    end
    if configuration.adjustment !== nothing
        config_dict[:adjustment] = to_dict(configuration.adjustment)
    end
    return config_dict
end

to_dict(x) = x

to_dict(v::AbstractVector) = [to_dict(x) for x in v]

from_dict!(x) = x

from_dict!(v::AbstractVector) = [from_dict!(x) for x in v]

"""
    from_dict!(d::Dict)

Converts a dictionary to a TMLE struct.
"""
function from_dict!(d::Dict{T, Any}) where T
    haskey(d, T(:type)) || return Dict(key => from_dict!(val) for (key, val) in d)
    constructor = eval(Meta.parse(pop!(d, :type)))
    return constructor(;[key => from_dict!(val) for (key, val) in d]...)
end

function read_yaml end
function write_yaml end
function read_json end
function write_json end