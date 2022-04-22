"""
Abstract type for callbacks that take a machine as an argument
"""
abstract type MachineCallback end

function after_machine_fit(::MachineCallback, ::Machine, id::Symbol) end
function finalize(estimation_report::NamedTuple, callback::MachineCallback) end

mutable struct MachineReportBuilder <: MachineCallback
    state::NamedTuple
    MachineReportBuilder() = new(NamedTuple{}())
end

after_machine_fit(callback::MachineReportBuilder, mach::Machine, id::Symbol) =
    callback.state = merge(callback.state, NamedTuple{(id,)}([mach]))

function finalize(estimation_report::NamedTuple, callback::MachineReportBuilder)
    Qmachs = Machine[]
    Fmachs = Machine[]
    println(estimation_report)
    for key in keys(callback.state)
        if occursin("Q", string(key))
            push!(Qmachs, getfield(callback.state, key))
        elseif occursin("F", string(key))
            
        end
    end

    machines = (
        Encoder = callback.state.Encoder,
        G       = callback.state.G,
        Q       = Qmachs,
        F       = Fmachs
    )
    return (estimation_report..., machines=machines)
end