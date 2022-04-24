"""
Abstract type for callbacks used during the TMLE procedure. Currently, it is associated with 
two events:
    - `after_fit`: which is triggered after every machine fit in the procedure
    - `after_tmle`: which is triggered after every single `TMLEReport` has been 
    built after the TMLE step.
"""
abstract type Callback end

function fit_with_callbacks!(mach, callbacks, verbosity, id)
    fit!(mach, verbosity=verbosity)
    after_fit(callbacks, mach, id)
end

"""
    after_fit(::Callback, ::Machine, ::Symbol)

This event is triggered after every `MLJBase.machine` `fit!` operation. It takes as
input the machine and an id to identify it.
"""
function after_fit(::Callback, ::Machine, ::Symbol) end
function after_fit(callbacks, mach::Machine, id::Symbol)
    for callback in callbacks
        after_fit(callback, mach, id)
    end
end

"""
    after_tmle(::Callback, ::TMLEReport, ::Int, ::Int)

This event is triggered after every `TMLRReport` has been built at the end of a the TMLE step. 
It takes as input the `TMLRReport` object, the target id and the query id to identify it.
"""
function after_tmle(::Callback, report::TMLEReport, target_id::Int, query_id::Int) end
function after_tmle(callbacks, report::TMLEReport, target_id::Int, query_id::Int)
    for callback in callbacks
        after_tmle(callback, report, target_id, query_id)
    end
end

function finalize(callbacks, estimation_report)
    for callback in callbacks
        estimation_report = finalize(callback, estimation_report)
    end
    return estimation_report
end
finalize(callback::Callback, estimation_report::NamedTuple) = estimation_report


"""
    MachineReporter()

Callback used to report all the fitted machines used during the TMLE procedure.
It is triggered on the `after_fit` event.
"""
mutable struct MachineReporter <: Callback
    state::Dict
    MachineReporter() = new(Dict())
end

after_fit(callback::MachineReporter, mach::Machine, id::Symbol) =
    callback.state[id] = mach

function finalize(callback::MachineReporter, estimation_report::NamedTuple)
    Qmachs = Machine[]
    Fmachs = Dict()

    for (key, mach) in callback.state
        keystring = string(key)
        if occursin("Q", keystring)
            push!(Qmachs, mach)
        elseif occursin("F", keystring)
            _, target_id, query_id = split(keystring, "_")
            target_id = parse(Int, target_id)
            query_id = parse(Int, query_id)
            Fmachs[((target_id), query_id)] = mach
        end
    end

    machines = (
        Encoder = callback.state[:Encoder],
        G       = callback.state[:G],
        Q       = Qmachs,
        F       = Fmachs
    )
    return (estimation_report..., machines=machines)
end


"""
    Reporter()

Callback used to report the estimation Report objects created during the TMLE procedure.
"""
mutable struct Reporter <: Callback
    state::Dict
    Reporter() = new(Dict())
end

after_tmle(callback::Reporter, report::TMLEReport, target_id::Int, query_id::Int) =
    callback.state[(target_id, query_id)] = report

finalize(callback::Reporter, estimation_report::NamedTuple) = 
    (estimation_report..., tmlereports=callback.state)

"""
A callback to save the TMLE estimation results to disk instead of keeping them in memory
"""
mutable struct JLD2Saver <: Callback
    file::JLD2.JLDFile
    save_machines::Bool
    JLD2Saver(filepath::String, save_machines::Bool=false) = new(jldopen(filepath, "w"), save_machines)
end

function after_tmle(callback::JLD2Saver, report::TMLEReport, target_id::Int, query_id::Int)
    if !haskey(callback.file, "tmlereports")
        group = JLD2.Group(callback.file, "tmlereports")
    else
        group = callback.file["tmlereports"]
    end
    group[string(target_id, "_", query_id)] = report
end

function after_fit(callback::JLD2Saver, mach::Machine, id::Symbol)
    if callback.save_machines
        if !haskey(callback.file, "machines")
            group = JLD2.Group(callback.file, "machines")
        else
            group = callback.file["machines"]
        end
        group[string(id)] = mach
    end
end

function finalize(callback::JLD2Saver, estimation_report::NamedTuple)
    callback.file["low_propensity_scores"] = estimation_report.low_propensity_scores
    close(callback.file)
    return estimation_report
end
