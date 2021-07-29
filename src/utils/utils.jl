
################################################################################
## Loss utility functions

function vector_mse(x, x̂)
    res = zero(eltype(x[1]))
    @inbounds for i in eachindex(x)
        res += sum((x[i] .- x̂[i]).^2)
    end
    # divide per number of time steps and batch size
    res /= length(x)*length(x[1][1,:])
    return res
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

function vector_kl(μ::T, logσ²::T) where T <: Matrix
    P = eltype(μ)
    s = zero(P)
    # go through initial conditions
    @inbounds for k in eachindex(μ)
        s += kl(μ[k], logσ²[k])
    end
    # divide per batch size
    s /= size(μ, 2)
    return s
end

## annealing factor scheduler
# based on https://github.com/haofuml/cyclical_annealing
function frange_cycle_linear(n_iter, start::T=0.0f0, stop::T=1.0f0,  n_cycle=4, ratio=0.5) where T
    L = ones(n_iter) * stop
    period = n_iter/n_cycle
    step = T((stop-start)/(period*ratio)) # linear schedule

    for c in 0:n_cycle-1
        v, i = start, 1
        while (v ≤ stop) & (Int(round(i+c*period)) < n_iter)
            L[Int(round(i+c*period))] = v
            v += step
            i += 1
        end
    end
    return T.(L)
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