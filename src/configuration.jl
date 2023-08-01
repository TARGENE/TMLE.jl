Configurations.from_dict(::Type{<:Estimand}, ::Type{Symbol}, s) = Symbol(s)
Configurations.from_dict(::Type{<:Estimand}, ::Type{NamedTuple}, s) = eval(Meta.parse(s))

function param_to_dict(Ψ::Estimand)
    d = to_dict(Ψ, YAMLStyle)
    d["type"] = typeof(Ψ)
    return d
end

"""
    estimands_from_yaml(filepath)

Read estimation `Estimands` from YAML file.
"""
function estimands_from_yaml(filepath)
    yaml_dict = YAML.load_file(filepath; dicttype=Dict{String, Any})
    params_dicts = yaml_dict["Estimands"]
    nparams = size(params_dicts, 1)
    estimands = Vector{Estimand}(undef, nparams)
    for index in 1:nparams
        d = params_dicts[index]
        type = pop!(d, "type")
        estimands[index] = from_dict(eval(Meta.parse(type)), d)
    end
    return estimands
end

"""
    estimands_to_yaml(filepath, estimands::Vector{<:Estimand})

Write estimation `Estimands` to YAML file.
"""
function estimands_to_yaml(filepath, estimands)
    d = Dict("Estimands" => [param_to_dict(Ψ) for Ψ in estimands])
    YAML.write_file(filepath, d)
end