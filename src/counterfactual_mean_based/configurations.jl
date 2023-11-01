Configurations.from_dict(::Type{<:StatisticalCMCompositeEstimand}, ::Type{Symbol}, s) = Symbol(s)
Configurations.from_dict(::Type{<:StatisticalCMCompositeEstimand}, ::Type{<:Tuple}, s) = eval(Meta.parse(s))
Configurations.from_dict(::Type{<:StatisticalCMCompositeEstimand}, ::Type{NamedTuple}, s) = eval(Meta.parse(s))

function estimand_to_dict(Ψ::StatisticalCMCompositeEstimand)
    d = to_dict(Ψ, YAMLStyle)
    d["type"] = typeof(Ψ)
    return d
end

"""
    estimands_from_yaml(filepath)

Read estimands from a YAML file.
"""
function estimands_from_yaml(filepath)
    yaml_dict = YAML.load_file(filepath; dicttype=Dict{String, Any})
    estimands_dicts = yaml_dict["Estimands"]
    nparams = size(estimands_dicts, 1)
    estimands = Vector(undef, nparams)
    for index in 1:nparams
        d = estimands_dicts[index]
        type = pop!(d, "type")
        estimands[index] = from_dict(eval(Meta.parse(type)), d)
    end
    return estimands
end

"""
    estimands_to_yaml(filepath, estimands)

Write estimands to a YAML file.
"""
function estimands_to_yaml(filepath, estimands)
    d = Dict("Estimands" => [estimand_to_dict(Ψ) for Ψ in estimands])
    YAML.write_file(filepath, d)
end