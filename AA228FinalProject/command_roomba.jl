
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

using DiscreteValueIteration


m = RoombaMDP()
mutable struct ToEnd <: Policy
    ts::Int64 # to track the current time-step.
end

# first seed the environment
Random.seed!(2)

# reset the policy
p = ToEnd(0) # here, the argument sets the time-steps elapsed to 0

# run the simulation
c = @GtkCanvas()
win = GtkWindow(c, "Roomba Environment", 600, 600)

# extract goal for heuristic controller
goal_xy = get_goal_pos(m)

function POMDPs.action(p::ToEnd, s::RoombaState)

    # go right for the first 25 time-steps
    if p.ts < 25
        p.ts += 1
        return RoombaAct(0.) # all actions are of type RoombaAct(theta)
    end
    p.ts += 1

    # compute the difference between our current heading and one that would
    # point to the goal
    goal_x, goal_y = goal_xy
    x,y = s
    ang_to_goal = atan(goal_y - y, goal_x - x)
    del_angle = wrap_to_pi(ang_to_goal)

    return RoombaAct(del_angle)
end

#
# for (t, step) in enumerate(stepthrough(m, p, max_steps=100))
#     @guarded draw(c) do widget
#
#         # the following lines render the room, the particles, and the roomba
#         ctx = getgc(c)
#         set_source_rgb(ctx,1,1,1)
#         paint(ctx)
#         render(ctx, m, step)
#
#         # render some information that can help with debugging
#         # here, we render the time-step, the state, and the observation
#         move_to(ctx,300,400)
#         show_text(ctx, @sprintf("t=%d, state=%s",t,string(step.s)))
#     end
#     show(c)
#     sleep(0.1) # to slow down the simulation
# end

# for ease of access...
m = RoombaMDP()

# now do value iteration
solver = ValueIterationSolver(max_iterations=100, belres=1e-3)

# initialize the policy by passing in your problem
policy = ValueIterationPolicy(m)

# solve for an optimal policy
# if verbose=false, the text output will be supressed (false by default)
policy_p = solve(solver, m)

is = RoombaState(-24, -20, 0, 3.)
a = action(policy_p, is)
print(a)


c = @GtkCanvas()
win = GtkWindow(c, "Roomba Environment", 600, 600)


for i = 1:1
    traj_rewards = 0
    init_state = nothing
    saved = []
    for (t, step) in enumerate(stepthrough(m, policy_p, max_steps=100))
        push!(saved, step[:sp])
        @guarded draw(c) do widget
            if t == 1
                init_state = step.s
            end
            # the following lines render the room, the particles, and the roomba
            ctx = getgc(c)
            set_source_rgb(ctx,1,1,1)
            paint(ctx)
            render(ctx, m, step)

            # render the goal
            gx, gy = transform_coords(goal_xy)
            set_source_rgba(ctx, 0.0, 0.0, 1.0, 1.0)
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
        end
        println(step.a, step.s.x, step.s.y)
        traj_rewards += step.r
        show(c)
        sleep(0.1) # to slow down the simulation
    end
    print(traj_rewards)
end



using Statistics

total_rewards = []

for exp = 1:100
    #println(string(exp))

    Random.seed!(exp)

    p = ToEnd(0)
    traj_rewards = sum([step.r for step in stepthrough(m,policy_p,max_steps=100)])

    push!(total_rewards, traj_rewards)
end

@printf("Mean Total Reward: %.3f, StdErr Total Reward: %.3f", mean(total_rewards), std(total_rewards)/sqrt(5))



println("end of file")

# do some stats stuff - run value iteration a bunch of times

# for (t, step) in enumerate(stepthrough(m, p, belief_updater, max_steps=100))
#     @guarded draw(c) do widget
#
#         # the following lines render the room, the particles, and the roomba
#         ctx = getgc(c)
#         set_source_rgb(ctx,1,1,1)
#         paint(ctx)
#         render(ctx, m, step)
#
#         # render some information that can help with debugging
#         # here, we render the time-step, the state, and the observation
#         move_to(ctx,300,400)
#         show_text(ctx, @sprintf("t=%d, state=%s, o=%.3f",t,string(step.s),step.o))
#     end
#     show(c)
#     sleep(0.1) # to slow down the simulation
# end
#
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
#
#
# #############
#
# using POMDPModels, BasicPOMCP
# solver = POMCPSolver()
# planner = solve(solver, m)
#
# for (s, a, o) in stepthrough(m, planner, "sao", max_steps=100)
#     println("State was $s,")
#     println("action $a was taken,")
#     println("and observation $o was received.\n")
# end
#
# for exp = 1:5
#     println(string(exp))
#
#     Random.seed!(exp)
#
#     p = ToEnd(0)
#     traj_rewards = sum([step.r for step in stepthrough(m,planner, max_steps=100)])
#
#     push!(total_rewards, traj_rewards)
# end
