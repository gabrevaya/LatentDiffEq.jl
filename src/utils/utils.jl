
################################################################################
## Loss utility functions

function vector_mse(x, x̂)
    res = zero(eltype(x[1]))
    @inbounds for i in eachindex(x)
        res += mean((x[i] .- x̂[i]).^2)
    end
    res /= length(x)
end

kl(μ, logσ²) = -logσ²/2f0 + ((exp(logσ²) + μ^2)/2f0) - 0.5f0

function vector_kl(μ::T, logσ²::T) where T <: Tuple{Matrix, Matrix}
    P = eltype(μ[1])
    s = zero(P)
    # go through initial conditions and parameters
    @inbounds for i in 1:2
        s1 = zero(P)
        @inbounds for k in eachindex(μ[i])
            s1 += kl(μ[i][k], logσ²[i][k])
        end
        # divide per batch size
        s1 /= size(μ[i], 2)
        s += s1
    end
    return s
end

function vector_kl(μ::T, logσ²::T) where T <: Tuple{Matrix}
    P = eltype(μ[1])
    s = zero(P)
    # go through initial conditions
    @inbounds for k in eachindex(μ[1])
        s += kl(μ[1][k], logσ²[1][k])
    end
    # divide per batch size
    s /= size(μ[1], 2)
    return s
end

## annealing factor scheduler
# start_af: start value of annealing factor
# end_af: end value of annealing factor
# ae: annealing epochs - after these epochs, the annealing factor is set to the end value.
# epoch: current epoch
# mb_id: current number of mini batch
# mb_amount: amount of mini batches
function annealing_factor(start_af, end_af, ae, epoch, mb_id, mb_amount)

    if ae > 0
        if epoch < ae
            return Float32(start_af + (end_af - start_af) * (mb_id + epoch * mb_amount) / (ae * mb_amount))
        else
            return Float32(end_af)
        end
    else
        return 1.0f0
    end

end


################################################################################
## Data pre-processing

function normalize_to_unit_segment(X)
    min_val = minimum(X)
    max_val = maximum(X)

    X̂ = (X .- min_val) ./ (max_val - min_val)
    return X̂, min_val, max_val
end

denormalize_unit_segment(X̂, min_val, max_val) = X̂ .* (max_val .- min_val) .+ min_val


################################################################################
## Training help function

function time_loader(x, full_seq_len, seq_len)

    x_ = Array{Float32, 3}(undef, (size(x,1), seq_len, size(x,3)))
    idxs = rand_time(full_seq_len, seq_len)
    for i in 1:size(x,3)
        x_[:,:,i] = x[:, idxs, i]
    end
    return Flux.unstack(x_, 2)
end

function rand_time(full_seq_len, seq_len)
    start_time = rand(1:full_seq_len - seq_len)
    idxs = start_time:start_time+seq_len-1
    return idxs
end