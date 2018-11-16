immutable SkierObsDist{G<:Function}
    p::PowseekerPOMDP{G}
    gps::Bool
    s::SkierState
    sp::SkierState
end

function rand(rng::AbstractRNG, d::SkierObsDist)
    if d.gps
        return SkierObs(d.sp.time, d.sp.pos + d.p.gps_std*randn(rng, 2), d.sp.psi + d.p.compass_std*randn(rng))
    else
        true_dist = norm(d.sp.pos-d.s.pos) 
        dist = true_dist*(1.0 + d.p.dist_std_frac*randn(rng))
        grad_local = true_grad_local(d)
        grad_obs = grad_local + d.p.grad_std*randn(rng, 2)
        return SkierObs(d.sp.time, dist, grad_obs)
    end
end

function true_grad_local(d::SkierObsDist)
    grad_global = mdp(d.p).gradient(d.sp.pos)
    psi = d.sp.psi
    C = @SMatrix [cos(psi) sin(psi); -sin(psi) cos(psi)]
    grad_local = C*grad_global
end

function pdf(d::SkierObsDist, o::SkierObs)
    if d.gps
        meas = get(o.pos) # should error if something is inconsistent
        meas_pdf = exp(-sum((meas - d.sp.pos).^2)/(2*d.p.gps_std^2))
        dir = get(o.dir)
        dir_diff = abs(dir - d.sp.psi)
        while dir_diff > pi
            dir_diff -= 2*pi
        end
        dir_diff = abs(dir_diff)
        dir_pdf = exp(-dir_diff^2/(2*d.p.compass_std^2))
        return meas_pdf*dir_pdf
    else
        true_dist = norm(d.sp.pos-d.s.pos)
        dist_pdf = exp(-(o.dist-true_dist)^2/(2*(true_dist*d.p.dist_std_frac)^2))
        true_grad = true_grad_local(d)
        grad_pdf = exp(-sum((o.grad-true_grad).^2)/(2*d.p.grad_std^2))
        return dist_pdf*grad_pdf
    end
end
