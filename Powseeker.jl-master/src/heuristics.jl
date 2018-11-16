immutable Downhill{G<:Function} <: Policy
    gradient::G
end
Downhill(p::PowseekerProblem) = Downhill(mdp(p).gradient)

function action(p::Downhill, s::SkierState)
    grad = p.gradient(s.pos)
    steepest = atan2(-grad[2], -grad[1])
    return steepest - s.psi
end

immutable Topout{G<:Function} <: Policy
    mdp::PowseekerMDP{G}
    tol::Float64
end
Topout(p::PowseekerProblem, tol::Float64) = Topout(mdp(p), tol)

function action(p::Topout, s::SkierState)
    grad = p.mdp.gradient(s.pos)
    if s.time >= p.mdp.duration || norm(grad) < p.tol # topped out
        return action(Downhill(p.mdp.gradient), s)
    else
        up = atan2(grad[2], grad[1])
        return up-s.psi
    end
end

immutable SkiOver{G<:Function} <: Policy
    mdp::PowseekerMDP{G}
end
SkiOver(p::PowseekerProblem) = SkiOver(mdp(p))

function action(p::SkiOver, s::SkierState)
    grad = p.mdp.gradient(s.pos)
    if s.time >= p.mdp.duration
        return action(Downhill(p.mdp.gradient), s)
    elseif dot(grad, [cos(s.psi), sin(s.psi)]) > 0 # facing uphill
        up = atan2(grad[2], grad[1])
        return up-s.psi
    else
        down = atan2(-grad[2], -grad[1])
        return down-s.psi
    end
end

immutable RandomlyCheckGPS{P<:Policy, RNG<:AbstractRNG} <: Policy
    p::P
    p_check::Float64
    rng::RNG
end

function action(p::RandomlyCheckGPS, b)
    check = rand(p.rng) < p.p_check
    if check
        return GPSOrAngle(true, 0.0)
    else
        s = rand(p.rng, b)
        return GPSOrAngle(false, action(p.p, s))
    end
end

#=
@with_kw immutable Center <: Policy
    rng::MersenneTwister = MersenneTwister(14)
end

function action(::Center, s::SkierState)
    center = atan2
end

function action(p::Center, c::ParticleCollection)
    s = rand(p.rng, c)
    return GPSOrAngle(false, action(p, s))
end
=#
