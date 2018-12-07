# Defines the environment as a POMDPs.jl MDP and POMDP
# maintained by {jmorton2,kmenda}@stanford.edu

# Wraps ang to be in (-pi, pi]
function wrap_to_pi(ang::Float64)
    if ang > pi
		ang -= 2*pi
	elseif ang <= -pi
		ang += 2*pi
    end
	ang
end

USE_COMMANDS = false

"""
State of a Roomba.

# Fields
- `x::Float64` x location in meters
- `y::Float64` y location in meters
- `status::Float64` indicator whether robot has reached goal state
- `cmd::Float64`
"""
@with_kw struct RoombaState <: FieldVector{4, Float64}
    x::Float64
    y::Float64
	status::Float64
	cmd::Float64
end

# Struct for a Roomba action
struct RoombaAct <: FieldVector{1, Float64}
    theta::Float64     # direction of movement
end

# action spaces
struct RoombaActions
	#actions::FieldVector{12, Float64}
end
# RoombaActions() = RoombaActions(collect(range(-pi, length=12, stop=pi)))


function gen_amap(aspace::RoombaActions)
    return nothing
end

function gen_amap(aspace::AbstractVector{RoombaAct})
    return Dict(aspace[i]=>i for i in 1:length(aspace))
end

"""
Define the Roomba MDP.

# Fields
- `v::Float64` constant velocity of Roomba [m/s]
- `dt::Float64` simulation time-step [s]
- `contact_pen::Float64` penalty for wall-contact
- `time_pen::Float64` penalty per time-step
- `goal_reward::Float64` reward for reaching goal
- `room::Room` environment room struct
- `sspace::SS` environment state-space (ContinuousRoombaStateSpace or DiscreteRoombaStateSpace)
- `aspace::AS` environment action-space struct
"""
@with_kw mutable struct RoombaMDP{SS,AS} <: MDP{RoombaState, RoombaAct}
    v::Float64  = 2  # m/s
    dt::Float64     = 0.5   # s
    contact_pen::Float64 = -1.0
    time_pen::Float64 = -0.1
    goal_reward::Float64 = 100
    room::Room  = Room()
    sspace::SS = DiscreteRoombaStateSpace(50, 40)
    aspace::AS = RoombaActions()
    _amap::Union{Nothing, Dict{RoombaAct, Int}} = gen_amap(aspace)
end

# state-space definitions
struct ContinuousRoombaStateSpace end

"""
Specify a DiscreteRoombaStateSpace
- `x_step::Float64` distance between discretized points in x
- `y_step::Float64` distance between discretized points in y
#- `cmd_step::Float64` distance between discretized points in command mappings # For Modified Problem
- `XLIMS::Vector` boundaries of room (x-dimension)
- `YLIMS::Vector` boundaries of room (y-dimension)

"""
struct DiscreteRoombaStateSpace
    x_step::Float64
    y_step::Float64
    XLIMS::Vector
    YLIMS::Vector
end

# function to construct DiscreteRoombaStateSpace:
# `num_x_pts::Int` number of points to discretize x range to
# `num_y_pts::Int` number of points to discretize y range to
## `num_cmd_pts::Int` number of points to discretize command mapping range to # For Modified Problem
#function DiscreteRoombaStateSpace(num_x_pts::Int, num_y_pts::Int)
function DiscreteRoombaStateSpace(num_x_pts::Int, num_y_pts::Int) # For Modified Problem

    # hardcoded room-limits
    # watch for consistency with env_room
    XLIMS = [-30.0, 20.0]
    YLIMS = [-30.0, 10.0]

    return DiscreteRoombaStateSpace((XLIMS[2]-XLIMS[1])/(num_x_pts-1),
                                    (YLIMS[2]-YLIMS[1])/(num_y_pts-1),
                                    XLIMS,YLIMS)
end



# Shorthands
const RoombaModel = RoombaMDP


# access the mdp of a RoombaModel
mdp(e::RoombaMDP) = e


# function to determine if there is contact with a wall
wall_contact(e::RoombaModel, state) = wall_contact(mdp(e).room, state[1:2])

# there is probably a prettier way to do this discrete action stuff
# POMDPs.actions(m::RoombaModel) = mdp(m).aspace
# POMDPs.n_actions(m::RoombaModel) = length(mdp(m).aspace)
POMDPs.actions(m::RoombaModel) = [RoombaAct(a) for a in range(-pi, length=25, stop=pi)[1:24]]
POMDPs.n_actions(m::RoombaModel) = 24

function POMDPs.actionindex(mdp::RoombaModel, action::RoombaAct)
	act = action[1]
	return argmin([abs(a - act) for a in range(-pi, length=25, stop=pi)[1:24]])
end;

# maps a RoombaAct to an index in a RoombaModel with discrete actions
# function POMDPs.actionindex(m::RoombaModel, a::RoombaAct)
#     if mdp(m)._amap != nothing
#         return mdp(m)._amap[a]
#     else
#         error("Action index not defined for continuous actions.")
#     end
# end

# function to get goal xy location for heuristic controllers
function get_goal_pos(m::RoombaModel)
    return mdp(m).room.goal_pos
end

function at_goal(x, y, m::RoombaModel)
    goal_x, goal_y = get_goal_pos(m)
	return (abs(x - goal_x) < 2) && abs(y - goal_y) < 2 # TODO: Make a more robust buffer
end

# initializes x,y of Roomba in the room
function POMDPs.initialstate(m::RoombaModel, rng::AbstractRNG)
    e = mdp(m)
    x, y = init_pos(e.room, rng)
	gx, gy = get_goal_pos(m)

	if USE_COMMANDS
		g_theta = atan(gy - y, gx - x)
		cmd =  argmin([abs(a - g_theta) for a in range(-pi, step=pi/2, stop=pi)])
		if cmd == 5
			cmd = 1
		end
	end

	while(at_goal(x,y,m))
		println("Goal pos: $(get_goal_pos(m)); Init pos: ($x, $y)")
		x, y = init_pos(e.room, rng)
	end

	if USE_COMMANDS
		is = RoombaState(x=x, y=y, cmd=cmd, status=0.0)
	else
		is = RoombaState(x=x, y=y, cmd=0, status=0.0)
	end
	#is = RoombaState(x=x, y=y, cmd_1=cmd_1, cmd_2=cmd_2, cmd_3=cmd_3, cmd_4=cmd_4 status=0.0)

    if mdp(m).sspace isa DiscreteRoombaStateSpace
        isi = state_index(m, is)
        is = index_to_state(m, isi)
    end

    return is
end

# transition Roomba state given curent state and action
function POMDPs.transition(m::RoombaModel,
                           s::AbstractVector{Float64},
                           a::AbstractVector{Float64})

    e = mdp(m)
    theta = a[1]
    theta = wrap_to_pi(theta)

    # propagate dynamics without wall considerations
    x, y = s
    dt = e.dt
	v = e.v

    # make sure we arent going through a wall
    p0 = SVector(x, y)
    heading = SVector(cos(theta), sin(theta))
    des_step = v*dt
    next_x, next_y = legal_translate(e.room, p0, heading, des_step)

    # Determine whether goal has been reached
    next_status = 1.0*at_goal(next_x, next_y, m)

	# Determine next command
	if USE_COMMANDS
		gx, gy = get_goal_pos(m)
		g_theta = atan(gy - y, gx - x)
		cmd =  argmin([abs(a - g_theta) for a in range(-pi, step=pi/2, stop=pi)])
		if cmd == 5
			cmd = 1
		end
		sp = RoombaState(x=next_x, y=next_y, cmd=cmd, status=next_status)
	else
		sp = RoombaState(x=next_x, y=next_y, cmd=0, status=next_status)
	end
    # define next state


    if mdp(m).sspace isa DiscreteRoombaStateSpace
        # round the states to nearest grid point
        si = state_index(m, sp)
        sp = index_to_state(m, si)
    end

    return Deterministic(sp)
end

# enumerate all possible states in a DiscreteRoombaStateSpace
function POMDPs.states(m::RoombaModel)
    if mdp(m).sspace isa DiscreteRoombaStateSpace
        ss = mdp(m).sspace
        x_states = range(ss.XLIMS[1], stop=ss.XLIMS[2], step=ss.x_step)
        y_states = range(ss.YLIMS[1], stop=ss.YLIMS[2], step=ss.y_step)
		cmd_states = [1., 2., 3., 4.]
        statuses = [0.,1.]
		if USE_COMMANDS
			return vec(collect(RoombaState(x=x,y=y,cmd=cmd,status=st) for x in x_states, y in y_states, cmd in cmd_states, st in statuses))
		else
			return vec(collect(RoombaState(x=x,y=y,cmd=0,status=st) for x in x_states, y in y_states, st in statuses))
		end
    else
        return mdp(m).sspace
    end
end

# return the number of states in a DiscreteRoombaStateSpace
function POMDPs.n_states(m::RoombaModel)
    if mdp(m).sspace isa DiscreteRoombaStateSpace
        ss = mdp(m).sspace
    	# 	nstates = prod((convert(Int, diff(ss.XLIMS)[1]/ss.x_step)+1,
        #                     convert(Int, diff(ss.YLIMS)[1]/ss.y_step)+1,
        #                     2))
		n_comms = 1
		if USE_COMMANDS
			n_comms = 4
		end

        nstates = prod((convert(Int, diff(ss.XLIMS)[1]/ss.x_step)+1,  # For Modified Problem
						convert(Int, diff(ss.YLIMS)[1]/ss.y_step)+1,
						2, n_comms))
        return nstates
    else
        error("State-space must be DiscreteRoombaStateSpace.")
    end
end

function POMDPs.stateindex(m::RoombaModel, s::RoombaState)
	return state_index(m, s)
end


# map a RoombaState to an index in a DiscreteRoombaStateSpace
function POMDPs.state_index(m::RoombaModel, s::RoombaState)
    if mdp(m).sspace isa DiscreteRoombaStateSpace
        ss = mdp(m).sspace
        xind = floor(Int, (s[1] - ss.XLIMS[1]) / ss.x_step + 0.5) + 1
        yind = floor(Int, (s[2] - ss.YLIMS[1]) / ss.y_step + 0.5) + 1
        stind = convert(Int, s[3] + 1)
		cmdind = convert(Int, s[4])
		# cmd1ind = floor(Int, (s[4] - (-pi)) / ss.cmd_step + 0.5) + 1  # For Modified Problem
		# cmd2ind = floor(Int, (s[5] - (-pi)) / ss.cmd_step + 0.5) + 1
		# cmd3ind = floor(Int, (s[6] - (-pi)) / ss.cmd_step + 0.5) + 1
		# cmd4ind = floor(Int, (s[7] - (-pi)) / ss.cmd_step + 0.5) + 1

        # lin = LinearIndices((convert(Int, diff(ss.XLIMS)[1]/ss.x_step)+1,
        #                     convert(Int, diff(ss.YLIMS)[1]/ss.y_step)+1,
        #                     2))
		n_comms = 1
		if USE_COMMANDS
			n_comms = 4
		end

		lin = LinearIndices((convert(Int, diff(ss.XLIMS)[1]/ss.x_step)+1,  # For Modified Problem
	                        convert(Int, diff(ss.YLIMS)[1]/ss.y_step)+1,
	                        2, n_comms))
		#return lin[xind,yind,stind]
		if USE_COMMANDS
			return lin[xind,yind,stind,cmdind]  # For Modified Problem
		else
			return lin[xind, yind, stind, 1] # only one "command"
		end
    else
        error("State-space must be DiscreteRoombaStateSpace.")
    end
end

# map an index in a DiscreteRoombaStateSpace to the corresponding RoombaState
function index_to_state(m::RoombaModel, si::Int)
    if mdp(m).sspace isa DiscreteRoombaStateSpace
        ss = mdp(m).sspace
        # lin = CartesianIndices((convert(Int, diff(ss.XLIMS)[1]/ss.x_step)+1,
        #                     convert(Int, diff(ss.YLIMS)[1]/ss.y_step)+1,
        #                     2))
		n_comms = 1
		if USE_COMMANDS
			n_comms = 4
		end
		lin = CartesianIndices((convert(Int, diff(ss.XLIMS)[1]/ss.x_step)+1,  # For Modified Problem
		                    convert(Int, diff(ss.YLIMS)[1]/ss.y_step)+1,
	                      	2, n_comms))

        #xi,yi,sti = Tuple(lin[si])
		xi,yi,sti,cmdi = Tuple(lin[si])  # For Modified Problem

        x = ss.XLIMS[1] + (xi-1) * ss.x_step
        y = ss.YLIMS[1] + (yi-1) * ss.y_step
        st = sti - 1
		cmd = cmdi

        return RoombaState(x=x, y=y, status=st, cmd=cmd)

    else
        error("State-space must be DiscreteRoombaStateSpace.")
    end
end


# defines reward function R(s,a,s')
function POMDPs.reward(m::RoombaModel,
                s::AbstractVector{Float64},
                a::AbstractVector{Float64},
                sp::AbstractVector{Float64})

    # penalty for each timestep elapsed
    cum_reward = mdp(m).time_pen

    # penalty for bumping into wall (not incurred for consecutive contacts)
    previous_wall_contact = wall_contact(m,s)
    current_wall_contact = wall_contact(m,sp)
    if(!previous_wall_contact && current_wall_contact)
        cum_reward += mdp(m).contact_pen
    end

    # terminal rewards
    cum_reward += mdp(m).goal_reward*(sp.status == 1.0)

    return cum_reward
end

# determine if a terminal state has been reached
POMDPs.isterminal(m::RoombaModel, s::AbstractVector{Float64}) = abs(s.status) > 0.0

# define discount factor
POMDPs.discount(m::RoombaModel) = 0.95

# struct to define an initial distribution over Roomba states
struct RoombaInitialDistribution{M<:RoombaModel}
    m::M
end

# definition of initialstate and initialstate_distribution for Roomba environment
POMDPs.rand(rng::AbstractRNG, d::RoombaInitialDistribution) = initialstate(d.m, rng)
POMDPs.initialstate_distribution(m::RoombaModel) = RoombaInitialDistribution(m)

# Render a room and show robot
function render(ctx::CairoContext, m::RoombaModel, step, saved)
    env = mdp(m)
    state = step[:sp]

    radius = ROBOT_W*6

    # render particle filter belief
    if haskey(step, :bp)
        bp = step[:bp]
        if bp isa AbstractParticleBelief
            for p in particles(bp)
                x, y = transform_coords(p[1:2])
                arc(ctx, x, y, radius, 0, 2*pi)
                set_source_rgba(ctx, 0.6, 0.6, 1, 0.3)
                fill(ctx)
            end
        end
    end

    # Render room
    render(env.room, ctx)

	# Render path
	prev = nothing
	for i in 1:length(saved)
		pos = saved[i]
		end_x, end_y = transform_coords(pos[1:2])
		if prev != nothing
			start_x, start_y = prev
			set_source_rgba(ctx, 1, 0.6, 0.6, 0.1+ 0.9(i/length(saved)))
			move_to(ctx, start_x, start_y)
		    line_to(ctx, end_x, end_y)
		    stroke(ctx)
		end
		prev = end_x, end_y
	end

    # Find center of robot in frame and draw circle
    x, y = transform_coords(state[1:2])
    arc(ctx, x, y, radius, 0, 2*pi)
    set_source_rgb(ctx, 1, 0.6, 0.6)
    fill(ctx)

    return ctx
end

function render(m::RoombaModel, step)
    io = IOBuffer()
    c = CairoSVGSurface(io, 800, 600)
    ctx = CairoContext(c)
    render(ctx, m, step)
    finish(c)
    return HTML(String(take!(io)))
end
