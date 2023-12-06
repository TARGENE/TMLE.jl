

mutable struct TreatmentTransformer <: MLJBase.Unsupervised
    encoder::OneHotEncoder
end

encoder() = OneHotEncoder(drop_last=true, ordered_factor=false)

"""
    TreatmentTransformer(;encoder=encoder())

Treatments in TMLE are represented by `CategoricalArrays`. If a treatment column
has type `OrderedFactor`, then its integer representation is used, make sure that 
the levels correspond to your expectations. All other columns are one-hot encoded.
"""
TreatmentTransformer(;encoder=encoder()) = TreatmentTransformer(encoder)

MLJBase.fit(model::TreatmentTransformer, verbosity::Int, X) =
    MLJBase.fit(model.encoder, verbosity, X)

function MLJBase.transform(model::TreatmentTransformer, fitresult, Xnew)
    Xt = MLJBase.transform(model.encoder, fitresult, Xnew)
    ordered_factors_names = []
    ordered_factors_values = []
    
    for colname in Tables.columnnames(Xnew)
        column = Tables.getcolumn(Xnew, colname)
        if eltype(scitype(column)) <: OrderedFactor
            try
                push!(ordered_factors_values, float(column)) 
                push!(ordered_factors_names, colname)
            catch e
                if isa(e, MethodError)
                    throw(ArgumentError("Incompatible categorical value's underlying type for column $colname, please convert to `<:Real`"))
                else
                    throw(e)
                end
            end
        end
    end
    ordered_factors_names = Tuple(ordered_factors_names)
    ordered_factors = NamedTuple{ordered_factors_names}(ordered_factors_values) 
    return merge(Tables.columntable(Xt), ordered_factors)
end

with_encoder(model; encoder=encoder()) = Pipeline(TreatmentTransformer(;encoder=encoder),  model)