
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

sensor = Command() # or Bumper() for the bumper version of the environment
m = RoombaPOMDP(sensor=sensor, mdp=RoombaMDP());

num_particles = 2000
# resampler = LidarResampler(num_particles, LowVarianceResampler(num_particles))
# resampler = BumperResampler(num_particles)
# for the bumper environment
resampler = CommandResampler(num_particles)

spf = SimpleParticleFilter(m, resampler)

theta_noise_coeff = 0.1

belief_updater = RoombaParticleFilter(spf, theta_noise_coeff);

# Define the policy to test
mutable struct ToEnd <: Policy
    ts::Int64 # to track the current time-step.
end

# extract goal for heuristic controller
goal_xy = get_goal_pos(m)

# define a new function that takes in the policy struct and current belief,
# and returns the desired action
function POMDPs.action(p::ToEnd, b::ParticleCollection{RoombaState})

    # spin around to localize for the first 25 time-steps
    if p.ts < 25
        p.ts += 1
        return RoombaAct(0.) # all actions are of type RoombaAct(v,om)
    end
    p.ts += 1

    # after 25 time-steps, we follow a proportional controller to navigate
    # directly to the goal, using the mean belief state

    # compute mean belief of a subset of particles
    s = mean(b)

    # compute the difference between our current heading and one that would
    # point to the goal
    goal_x, goal_y = goal_xy
    x,y = s
    ang_to_goal = atan(goal_y - y, goal_x - x)
    del_angle = wrap_to_pi(ang_to_goal)

    return RoombaAct(del_angle)
end

# first seed the environment
Random.seed!(2)

# reset the policy
p = ToEnd(0) # here, the argument sets the time-steps elapsed to 0

# run the simulation
c = @GtkCanvas()
win = GtkWindow(c, "Roomba Environment", 600, 600)

# is = RoombaState(x=-5.,y=-10.,status=0.0)
# dist = initialstate_distribution(m)
# b0 = initialize_belief(belief_updater, dist)

for (t, step) in enumerate(stepthrough(m, p, belief_updater, max_steps=100))
    @guarded draw(c) do widget

        # the following lines render the room, the particles, and the roomba
        ctx = getgc(c)
        set_source_rgb(ctx,1,1,1)
        paint(ctx)
        render(ctx, m, step)

        # render the goal
        gx, gy = transform_coords(goal_xy)
        set_source_rgba(ctx, 0.0, 0.0, 1.0, 1.0)
        arc(ctx, gx, gy, 15, 0, 2*pi)

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
    sleep(0.1) # to slow down the simulation
end

using Statistics

total_rewards = []

for exp = 1:5
    println(string(exp))

    Random.seed!(exp)

    p = ToEnd(0)
    traj_rewards = sum([step.r for step in stepthrough(m,p,belief_updater, max_steps=100)])

    push!(total_rewards, traj_rewards)
end

@printf("Mean Total Reward: %.3f, StdErr Total Reward: %.3f", mean(total_rewards), std(total_rewards)/sqrt(5))


#############

using POMDPModels, BasicPOMCP
solver = POMCPSolver()
planner = solve(solver, m)

for (s, a, o) in stepthrough(m, planner, "sao", max_steps=100)
    println("State was $s,")
    println("action $a was taken,")
    println("and observation $o was received.\n")
end

for exp = 1:5
    println(string(exp))

    Random.seed!(exp)

    p = ToEnd(0)
    traj_rewards = sum([step.r for step in stepthrough(m,planner, max_steps=100)])

    push!(total_rewards, traj_rewards)
end
