@recipe function f(p::PowseekerProblem)
    n = 100
    x = linspace(-4000, 4000, n)
    y = linspace(-4000, 4000, n)
    ratio --> :equal
    seriestype --> :contour
    x, y, mdp(p).topology
end

@recipe function f(s::SkierState)
    len = 0.2
    u = len*cos(s.psi)
    v = len*sin(s.psi)
    seriestype --> :quiver
    quiver --> ([u], [v])
    [s.pos[1]], [s.pos[2]]
end

@recipe function f(v::AbstractVector{SkierState})
    len = 200
    psis = collect(s.psi for s in v)
    xs = collect(s.pos[1] for s in v)
    ys = collect(s.pos[2] for s in v)
    us = len*cos.(psis)
    vs = len*sin.(psis)
    @series begin
        label --> "True Path"
        xs, ys
    end
    @series begin
        label --> "direction"
        seriestype --> :quiver
        quiver --> (us, vs)
        xs, ys
    end
end


@recipe function f(hist::AbstractMDPHistory{SkierState, Float64})
    @series begin
        state_hist(hist)
    end
    len = 200
    sh = state_hist(hist)[1:end-1]
    psis = collect(s.psi for s in sh)
    xs = collect(s.pos[1] for s in sh)
    ys = collect(s.pos[2] for s in sh)
    aus = len*cos(psis + collect(action_hist(hist)))
    avs = len*sin(psis + collect(action_hist(hist)))
    @series begin
        label --> "action"
        seriestype --> :quiver
        quiver --> (aus, avs)
        xs, ys
    end
end

@recipe function f(b::ParticleCollection{SkierState})
    v = particles(b)
    len = 50
    psis = collect(s.psi for s in v)
    xs = collect(s.pos[1] for s in v)
    ys = collect(s.pos[2] for s in v)
    us = len*cos.(psis)
    vs = len*sin.(psis)
    @series begin
        color --> "black"
        seriestype --> :quiver
        quiver := (us, vs)
        xs, ys
    end
end

@recipe function f(h::AbstractPOMDPHistory{SkierState, GPSOrAngle})
    @series begin
        state_hist(h)[1:end-1]
    end
    @series begin
        belief_hist(h)[end]
    end
    oh = observation_hist(h)[1:end-1]
    gps_x = [get(o.pos)[1] for o in oh if !isnull(o.pos)]
    gps_y = [get(o.pos)[2] for o in oh if !isnull(o.pos)]
    if !isempty(gps_x)
        @series begin
            seriestype := :scatter
            marker --> :circle
            label := "GPS Measurements"
            gps_x, gps_y
        end
    end
end
