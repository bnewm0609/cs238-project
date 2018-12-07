
# activate project environment
# include these lines of code in any future scripts/notebooks
#---
import Pkg
if !haskey(Pkg.installed(), "AA228FinalProject")
    jenv = joinpath(dirname(@__FILE__()), ".") # this assumes the notebook is in the same dir
    # as the Project.toml file, which should be in top level dir of the project.
    # Change accordingly if this is not the case.
    Pkg.activate(jenv)
end
#---

Pkg.instantiate()
# Pkg.install("Cairo")

# import necessary packages
using AA228FinalProject
using POMDPs
using POMDPPolicies
using BeliefUpdaters
using ParticleFilters
using POMDPSimulators
using Cairo
using Gtk
using Random
using Printf
using POMDPModels
using POMDPSimulators
using BasicPOMCP
using POMCPOW
using ARDESPOT
using MCVI
using FIB
using SARSOP
using QMDP

Random.seed!(4308) #28 is hard

sensor = Command()
m = RoombaPOMDP(sensor=sensor, mdp=RoombaMDP());
num_particles = 3000

resampler = CommandResampler(num_particles, LowVarianceResampler(num_particles))
spf = SimpleParticleFilter(m, resampler)

theta_noise_coeff = 0.1

belief_updater = RoombaParticleFilter(spf, theta_noise_coeff);

# Define the policy to test
mutable struct ToEnd <: Policy
    ts::Int64 # to track the current time-step.
end

# extract goal for heuristic controller
goal_xy = get_goal_pos(m)

# reset the policy
p = ToEnd(0) # here, the argument sets the time-steps elapsed to 0

# run the simulation
c = @GtkCanvas()

win = GtkWindow(c, "Roomba Environment", 600, 600)

struct Heuristic <:Policy
end
function POMDPs.action(p::Heuristic, s::RoombaState)
    x, y = s
    gx, gy = get_goal_pos(m)
    th_goal = atan(gy - y, gx - x)
    return RoombaAct(th_goal, true)
end
p = Heuristic()
solver = POMCPSolver(c=1., max_depth=200, estimate_value=FORollout(p))
p = solve(solver, m)


# solver = FIBSolver()
# p = solve(solver, m)
# solver = QMDPSolver(max_iterations=3,
                    # tolerance=1e-3,
                    # verbose=true
                   # )
# p = solve(solver, m)
# solver = SARSOPSolver()
# p = solve(solver, m)
# solver = DESPOTSolver(bounds=(0.0,10.))
# p = solve(solver, m)
solver = POMCPOWSolver(max_depth=50, criterion=MaxUCB(20.0))# ,
p = solve(solver, m)
# hr = HistoryRecorder(max_steps=100)
# hist = simulate(hr, m, p)
# for (s, b, a, r, sp, o) in hist
#     @show s, a, r, sp
# end

# is = RoombaState(x=-5.,y=-10.,status=0.0)
# dist = initialstate_distribution(m)
# b0 = initialize_belief(belief_updater, dist)

saved = []

for (t, step) in enumerate(stepthrough(m, p, belief_updater, max_steps=500))
    push!(saved, step[:sp])
    @guarded draw(c) do widget

        # the following lines render the room, the particles, and the roomba
        ctx = getgc(c)
        set_source_rgb(ctx,1,1,1)
        paint(ctx)
        render(ctx, m, step, saved)

        # render the goal
        gx, gy = transform_coords(goal_xy)
        set_source_rgba(ctx, 0.0, 1.0, 0.0, 1.0)
        arc(ctx, gx, gy, 15, 0, 2*pi)
        fill(ctx)

        # render some information that can help with debugging
        # here, we render the time-step, the state, and the observation
        move_to(ctx,70,40)
        set_source_rgba(ctx, 0.0, 0.0, 0.0, 1.0)
        show_text(ctx, @sprintf("t=%d",t))
        move_to(ctx,60,570)
        set_source_rgb(ctx, 1, 0.6, 0.6)
        show_text(ctx, @sprintf("x=%.3f, y=%.3f",step.s.x,step.s.y))
        move_to(ctx,60,580)
        show_text(ctx, @sprintf("obs=%s",step.o))
        arc(ctx, step.o*120, 500, 60, 0, 2*pi)
        set_source_rgba(ctx, 0.0, 1.0, 0.0, 0.1)
        fill(ctx)
    end
    show(c)
    sleep(0.01) # to slow down the simulation
end


# define a new function that takes in the policy struct and current belief,
# # and returns the desired action
# function POMDPs.action(p::ToEnd, b::ParticleCollection{RoombaState})
#
#     # spin around to localize for the first 25 time-steps
#     if p.ts < 25
#         p.ts += 1
#         return RoombaAct(0., true) # all actions are of type RoombaAct(v,om)
#     end
#     p.ts += 1
#
#     # after 25 time-steps, we follow a proportional controller to navigate
#     # directly to the goal, using the mean belief state
#
#     # compute mean belief of a subset of particles
#     s = mean(b)
#
#     # compute the difference between our current heading and one that would
#     # point to the goal
#     goal_x, goal_y = goal_xy
#     x,y = s
#     ang_to_goal = atan(goal_y - y, goal_x - x)
#     del_angle = wrap_to_pi(ang_to_goal)
#
#     return RoombaAct(del_angle, true)
# end

############

# using Statistics
#
# total_rewards = []
#
# for exp = 1:5
#     println(string(exp))
#
#     Random.seed!(exp)
#
#     p = ToEnd(0)
#     traj_rewards = sum([step.r for step in stepthrough(m,p,belief_updater, max_steps=100)])
#
#     push!(total_rewards, traj_rewards)
# end
#
# @printf("Mean Total Reward: %.3f, StdErr Total Reward: %.3f", mean(total_rewards), std(total_rewards)/sqrt(5))

#############
