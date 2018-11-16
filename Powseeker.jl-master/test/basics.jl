using Powseeker
using Plots
using POMDPs
using POMDPToolbox
using ForwardDiff


mdp = PowseekerMDP()
rng = MersenneTwister(100)

tries = [ 
            SkierState(1, [-550, -1150], deg2rad(60)),
            SkierState(1, [-50, 1400], deg2rad(270)),
            SkierState(1, [1500, 0], deg2rad(180)),
            SkierState(1, [-100, -100], deg2rad(225)),
            SkierState(1, [-100, -100], deg2rad(270)),
            SkierState(1, [-1600, 200], deg2rad(300)),
            SkierState(1, [-1600, 200], deg2rad(10)),
#             SkierState(1, [0, 1550], deg2rad(225)),
#             SkierState(1, [-350, -750], deg2rad(150)),
#             SkierState(1, [-350, -750], deg2rad(315)),
#             SkierState(1, [200, -1700], 0)
        ]

results = []
for s in tries
    n, r = generate_sr(mdp, s, 0.0, rng)
    @show r
    push!(results, n)
    @show mdp.topology(n.pos) - mdp.topology(s.pos)
end

plot(mdp)
for i in 1:length(tries)
    plot!([tries[i], results[i]])
end
gui()
