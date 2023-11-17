module JSONExt

using JSON
using TMLE

"""
    configuration_from_json(file)

Loads a YAML configuration file containing:
    - Estimands
    - An SCM (optional)
    - An Adjustment Method (optional)
"""
TMLE.configuration_from_json(file) = from_dict!(JSON.parsefile(file, dicttype=Dict{Symbol, Any}))

"""
    configuration_to_json(file, config::Configuration)

Writes a `Configuration` struct to a YAML file. The latter can be deserialized 
with `configuration_from_yaml`.
"""
function TMLE.configuration_to_json(file, config; indent=1)
    open(file, "w") do io 
        JSON.print(io, to_dict(config), indent)
    end
end

end