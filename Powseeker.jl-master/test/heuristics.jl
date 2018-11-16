using POMDPs
using POMDPToolbox
using Powseeker
using Plots

function plot_ro(policy::Policy)
    n = 100
    xs = linspace(-4000, 4000, 100)
    ys = linspace(-4000, 4000, 100)
    
    contour(xs, ys, (x,y)->simxy(policy, x, y), fill=true, title="$(typeof(policy))", zlim=(0,40))
end

function simxy(policy, x, y)
    mdp = PowseekerMDP()
    sim = RolloutSimulator(max_steps=mdp.duration, rng=MersenneTwister(1))
    s = SkierState(1, [x, y], deg2rad(90))
    return simulate(sim, mdp, policy, s)
end

mdp = PowseekerMDP()
plots = []
push!(plots, plot_ro(SkiOver(mdp)))
push!(plots, plot_ro(Downhill(mdp)))
push!(plots, plot_ro(Topout(mdp, 0.1)))
plot(plots...)
gui()
