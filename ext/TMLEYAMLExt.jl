module TMLEYAMLExt

using YAML
using TMLE

"""
    load_estimands_config(file)

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
TMLE.configuration_to_yaml(file, config::Configuration) = YAML.write_file(file, to_dict(config))

end