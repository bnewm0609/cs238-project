immutable SkierUnif
    time::Int
    xlim::NTuple{2, Float64}
    ylim::NTuple{2, Float64}
end

eltype(::Type{SkierUnif}) = SkierState

function rand(rng::AbstractRNG, d::SkierUnif)
    x = rand(rng)*(last(d.xlim)-first(d.xlim))+first(d.xlim)
    y = rand(rng)*(last(d.ylim)-first(d.ylim))+first(d.ylim)
    psi = rand(rng)*2*pi
    return SkierState(d.time, [x,y], psi)
end

initial_state_distribution(p::PowseekerProblem) = SkierUnif(1, mdp(p).xlim, mdp(p).ylim)
