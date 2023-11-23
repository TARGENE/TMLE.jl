module JSONExt

using JSON
using TMLE

"""
    read_json(file)

Loads a YAML configuration file containing:
    - Estimands
    - An SCM (optional)
    - An Adjustment Method (optional)
"""
TMLE.read_json(file) = TMLE.from_dict!(JSON.parsefile(file, dicttype=Dict{Symbol, Any}))

"""
    write_json(file, config::Configuration)

Writes a `Configuration` struct to a YAML file. The latter can be deserialized 
with `read_yaml`.
"""
function TMLE.write_json(file, config; indent=1)
    open(file, "w+") do io 
        write(io, JSON.json(TMLE.to_dict(config), indent))
    end
end

end