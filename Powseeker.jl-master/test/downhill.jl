using Powseeker
using POMDPs
using POMDPToolbox
using Plots

mdp = PowseekerMDP()
policy = Downhill(mdp)
hr = HistoryRecorder(max_steps=100)
is = SkierState(1, [1200, 0], 0)

hist = simulate(hr, mdp, policy, is)

gr()
plot(mdp)
plot!(hist)
gui()
