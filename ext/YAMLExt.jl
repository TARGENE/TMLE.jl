module YAMLExt

using YAML
using TMLE

"""
    read_yaml(file)

Loads a YAML configuration file containing:
    - Estimands
    - An SCM (optional)
    - An Adjustment Method (optional)
"""
TMLE.read_yaml(file) = TMLE.from_dict!(YAML.load_file(file, dicttype=Dict{Symbol, Any}))

"""
    write_yaml(file, config::Configuration)

Writes a `Configuration` struct to a YAML file. The latter can be deserialized 
with `read_yaml`.
"""
TMLE.write_yaml(file, config) = YAML.write_file(file, TMLE.to_dict(config))

end