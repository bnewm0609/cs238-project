# specification of particle filters for the bumper and lidar Roomba environments
# maintained by {jmorton2,kmenda}@stanford.edu

import POMDPs

# structs specifying resamplers for bumper and lidar sensors
struct BumperResampler
    n::Int # number of particles
end

struct LidarResampler
    n::Int # number of particles
    lvr::LowVarianceResampler
end

struct CommandResampler
    n::Int
end

struct OverlappingCommandResampler
    n::Int
end

"""
Definition of the particle filter for the Roomba environment
Fields:
- `spf::SimpleParticleFilter` standard particle filter struct defined in ParticleFilters.jl
- `theta_noise_coeff::Float64` coefficient to scale particle-propagation noise in theta
"""
mutable struct RoombaParticleFilter <: POMDPs.Updater
    spf::SimpleParticleFilter
    theta_noise_coefficient::Float64
end

# for regular command resampler, also have weights {0, 1}
function ParticleFilters.resample(cr::CommandResampler, b::WeightedParticleBelief{RoombaState}, rng::AbstractRNG)
    new = RoombaState[]
    for (p, w) in weighted_particles(b)
        if w == 1.0
            push!(new, p)
        else
            @assert w == 0
        end
    end
    if isempty(new) # no particles consistent with observations
        return ParticleCollection(particles(b))
    end
    extras = rand(rng, new, cr.n-length(new))
    for p in extras
        push!(new, p)
    end
    return ParticleCollection(new)
end

# Resample function for weights in {0,1} necessary for bumper sensor
function ParticleFilters.resample(br::BumperResampler, b::WeightedParticleBelief{RoombaState}, rng::AbstractRNG)
    new = RoombaState[]
    for (p, w) in weighted_particles(b)
        if w == 1.0
            push!(new, p)
        else
            @assert w == 0
        end
    end
    if isempty(new) # no particles consistent with observations
        return ParticleCollection(particles(b))
    end
    extras = rand(rng, new, br.n-length(new))
    for p in extras
        push!(new, p)
    end
    return ParticleCollection(new)
end

# resample function for unweighted particles
function ParticleFilters.resample(br::Union{BumperResampler,LidarResampler,
                                            CommandResampler,
                                            OverlappingCommandResampler},
                                 b, rng::AbstractRNG)
    ps = Array{RoombaState}(undef, br.n)
    for i in 1:br.n
        ps[i] = rand(rng, b)
    end
    return ParticleCollection(ps)
end

# Resample function for continuous weights necessary for lidar sensor
function ParticleFilters.resample(lr::LidarResampler, b::WeightedParticleBelief{RoombaState}, rng::AbstractRNG)
    ps = resample(lr.lvr, b, rng)
    return ps
end

# Resample function for weights with OverlappingCommand sensor - each command
# can refer to two directions
function ParticleFilters.resample(ocr::OverlappingCommandResampler, b::WeightedParticleBelief{RoombaState}, rng::AbstractRNG)
    newState = RoombaState[]
    for (p, w) in weighted_particles(b)
        if w <= 1.0 && w != 0
            push!(newState, p)
        else
            @assert w == 0
        end
    end
    if isempty(newState) # no particles consistent with observations
        return ParticleCollection(particles(b))
    end
    extras = rand(rng, newState, ocr.n-length(newState))
    for p in extras
        push!(newState, p)
    end
    return ParticleCollection(newState)
end
# Modified Update function adds noise to the actions that propagate particles
function POMDPs.update(up::RoombaParticleFilter, b::ParticleCollection{RoombaState}, a, o)
    ps = particles(b)
    pm = up.spf._particle_memory
    wm = up.spf._weight_memory
    resize!(pm, 0)
    resize!(wm, 0)
    sizehint!(pm, n_particles(b))
    sizehint!(wm, n_particles(b))
    all_terminal = true
    for i in 1:n_particles(b)
        s = ps[i]
        if !isterminal(up.spf.model, s)
            all_terminal = false
            # noise added here:
            a_pert = a + SVector(up.theta_noise_coefficient*(rand(up.spf.rng)-0.5))
            sp = generate_s(up.spf.model, s, a_pert, up.spf.rng)
            push!(pm, sp)
            push!(wm, obs_weight(up.spf.model, s, a_pert, sp, o))
        end
    end
    # if all particles are terminal, return previous belief state
    if all_terminal
        return b
    end

    return resample(up.spf.resample, WeightedParticleBelief{RoombaState}(pm, wm, sum(wm), nothing), up.spf.rng)
end

# initialize belief state
function ParticleFilters.initialize_belief(up::RoombaParticleFilter, d::Any)
    resample(up.spf.resample, d, up.spf.rng)
end
