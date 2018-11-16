module Powseeker

using StaticArrays
using ParticleFilters
using POMDPs
using Plots
using Parameters
using POMDPToolbox

import POMDPs: generate_sr, generate_s
import POMDPs: isterminal, initial_state_distribution, actions
import POMDPs: observation
import POMDPs: pdf
import POMDPs: action
import Base: rand, eltype

const Vec2 = SVector{2,Float64}

export
    PowseekerMDP,
    PowseekerPOMDP,
    SkierState,
    GPSOrAngle,
    SkierObs,
    SkierUnif,

    peaks,
    scaled_peaks,
    grad_peaks,
    grad_scaled_peaks,
    peaks2,
    scaled_peaks2,
    grad_peaks2,
    grad_scaled_peaks2,
    plot_peaks,
    plot_scaled_peaks,

    mdp,

    Downhill,
    Topout,
    SkiOver,
    RandomlyCheckGPS

include("peaks.jl")

# package code goes here
immutable SkierState
    time::Int
    pos::Vec2
    psi::Float64
end

immutable GPSOrAngle
    gps::Bool
    angle::Float64
end

immutable SkierObs
    time::Int
    pos::Nullable{Vec2}
    dir::Nullable{Float64}
    dist::Float64
    grad::Vec2

    SkierObs(t::Int, p::Vec2, dir::Float64) = new(t, Nullable{Vec2}(p), Nullable{Float64}(dir))
    SkierObs(t::Int, d::Float64, g::Vec2) = new(t, Nullable{Vec2}(), Nullable{Float64}(), d, g)
end

@with_kw immutable PowseekerMDP{G<:Function} <: MDP{SkierState, Float64}
    topology::Function      = scaled_peaks2
    gradient::G             = grad_scaled_peaks2
    duration::Int           = 60
    dt::Float64             = 2.0
    step_len::Float64       = 60.0
    force::Float64          = 4.0
    terminal_vel::Float64   = 50.0
    max_flat_speed::Float64 = 4.0
    psi_std::Float64        = deg2rad(0.5)
    vel_std::Float64        = 0.05
    xlim::NTuple{2,Float64} = (-4000.0, 4000.0)
    ylim::NTuple{2,Float64} = (-4000.0, 4000.0)
end

@with_kw immutable PowseekerPOMDP{G<:Function} <: POMDP{SkierState, GPSOrAngle, SkierObs}
    mdp::PowseekerMDP{G}    = PowseekerMDP()
    dist_std_frac::Float64  = 0.3
    grad_std::Float64       = 0.2
    gps_std::Float64        = 50.0
    compass_std::Float64    = deg2rad(10.0)
end

typealias PowseekerProblem Union{PowseekerMDP, PowseekerPOMDP}
mdp(p::PowseekerMDP) = p
mdp(p::PowseekerPOMDP) = p.mdp

function generate_sr(pp::PowseekerProblem, s::SkierState, a::Float64, rng::AbstractRNG)
    p = mdp(pp)
    dt = p.dt
    mfs = p.max_flat_speed
    vel = 0.0
    pos = s.pos
    psi = s.psi + a
    isteps = round(Int, p.step_len/dt)
    r = 0.0
    for i in 1:isteps
        dir = SVector(cos(psi), sin(psi))
        d = SkiLineDynamics(p, pos, dir)
        vec = rk4step(d, vel)
        vel = vec[1]+randn(rng)*p.vel_std
        pos += vec[2]*dir
        psi += randn(rng)*p.psi_std
        r += exp(vel)
        # r += vel^8
    end
    sp = SkierState(s.time+1, pos, psi)
    r /= isteps
    return (sp, r)
end

function generate_s(p::PowseekerProblem, s::SkierState, a::Float64, rng::AbstractRNG)
    sp, r = generate_sr(mdp(p), s, a, rng)
    return sp
end

function generate_sr(p::PowseekerPOMDP, s::SkierState, a::GPSOrAngle, rng::AbstractRNG)
    if a.gps
        return SkierState(s.time+1, s.pos, s.psi), 0.0
    else
        return generate_sr(mdp(p), s, a.angle, rng)
    end
end

function generate_s(p::PowseekerPOMDP, s::SkierState, a::GPSOrAngle, rng::AbstractRNG)
    if a.gps
        return SkierState(s.time+1, s.pos, s.psi)
    else
        return generate_s(mdp(p), s, a.angle, rng)
    end
end

function observation(p::PowseekerPOMDP, s::SkierState, a::GPSOrAngle, sp::SkierState)
    return SkierObsDist(p, a.gps, s, sp)
end

isterminal(p::PowseekerProblem, s::SkierState) = s.time > mdp(p).duration

include("runge-kutta.jl")
include("actions.jl")
include("observations.jl")
include("initial.jl")
include("heuristics.jl")
include("visualization.jl")

end # module
