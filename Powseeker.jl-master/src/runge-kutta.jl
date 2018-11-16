immutable SkiLineDynamics{G<:Function}
    p::PowseekerMDP{G}
    pos::Vec2
    dir::Vec2
end

function rk4step(d::SkiLineDynamics, vel::Float64)
    h = d.p.dt
    vec = SVector(vel, 0.0)
    k1 = f(d, vec)
    k2 = f(d, vec+h/2*k1)
    k3 = f(d, vec+h/2*k2)
    k4 = f(d, vec+h*k3)
    return vec + h/6*(k1+2*k2+2*k3+k4)
end

function f(d::SkiLineDynamics, vec::Vec2)
    vel = vec[1]
    dist = vec[2]
    h = d.p.dt
    mfs = d.p.max_flat_speed
    theta = atan(dot(d.p.gradient(d.pos+dist*d.dir), d.dir)) # slope angle
    acc = d.p.force*max(0.0, mfs-vel)/mfs - 9.8*sin(theta) - sign(vel)*9.8*(vel/d.p.terminal_vel)^2
    return SVector(acc, vel*cos(theta))
end
