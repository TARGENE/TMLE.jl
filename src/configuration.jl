Configurations.from_dict(::Type{<:Parameter}, ::Type{Symbol}, s) = Symbol(s)
Configurations.from_dict(::Type{<:Parameter}, ::Type{NamedTuple}, s) = eval(Meta.parse(s))

function param_to_dict(Ψ::Parameter)
    d = to_dict(Ψ, YAMLStyle)
    d["type"] = typeof(Ψ)
    return d
end

"""
    parameters_from_yaml(filepath)

Read estimation `Parameters` from YAML file.
"""
function parameters_from_yaml(filepath)
    yaml_dict = YAML.load_file(filepath; dicttype=Dict{String, Any})
    params_dicts = yaml_dict["Parameters"]
    nparams = size(params_dicts, 1)
    parameters = Vector{Parameter}(undef, nparams)
    for index in 1:nparams
        d = params_dicts[index]
        type = pop!(d, "type")
        parameters[index] = from_dict(eval(Meta.parse(type)), d)
    end
    return parameters
end

"""
    parameters_to_yaml(filepath, parameters::Vector{<:Parameter})

Write estimation `Parameters` to YAML file.
"""
function parameters_to_yaml(filepath, parameters)
    d = Dict("Parameters" => [param_to_dict(Ψ) for Ψ in parameters])
    YAML.write_file(filepath, d)
end