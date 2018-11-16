using Powseeker
using Plots

plot(PowseekerMDP())
plot!(SkierState(1, [0.0, 0.0], deg2rad(15)))
gui()
