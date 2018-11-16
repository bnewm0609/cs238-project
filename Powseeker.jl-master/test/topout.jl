using Powseeker
using POMDPs
using POMDPToolbox
using Plots

mdp = PowseekerMDP()
policy = Topout(mdp, 0.05)


hr = HistoryRecorder(max_steps=100)
# is = SkierState(1, [1200, 0], 0)
# is = SkierState(1, [-750, 1100], 0)
is = initial_state(mdp, Base.GLOBAL_RNG)
hist = simulate(hr, mdp, policy, is)
@show sum(reward_hist(hist))

plotlyjs()
plot(mdp)
plot!(hist)
gui()
