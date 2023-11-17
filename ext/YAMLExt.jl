module YAMLExt

using YAML
using TMLE

"""
    configuration_from_yaml(file)

Loads a YAML configuration file containing:
    - Estimands
    - An SCM (optional)
    - An Adjustment Method (optional)
"""
TMLE.configuration_from_yaml(file) = from_dict!(YAML.load_file(file, dicttype=Dict{Symbol, Any}))

"""
    configuration_to_yaml(file, config::Configuration)

Writes a `Configuration` struct to a YAML file. The latter can be deserialized 
with `configuration_from_yaml`.
"""
TMLE.configuration_to_yaml(file, config) = YAML.write_file(file, to_dict(config))

end