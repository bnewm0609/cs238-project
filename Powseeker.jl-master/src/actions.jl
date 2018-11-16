immutable AngleSpace end

rand(rng::AbstractRNG, ::AngleSpace) = rand(rng)*2*pi
actions(::PowseekerMDP) = AngleSpace()

immutable GPSOrAngleSpace
    p_gps::Float64
end
actions(::PowseekerPOMDP) = GPSOrAngleSpace(0.1)

function rand(rng::AbstractRNG, s::GPSOrAngleSpace)
    if rand(rng) < s.p_gps
        return GPSOrAngle(true, 0.0)
    else
        return GPSOrAngle(false, rand(rng, AngleSpace()))
    end
end
