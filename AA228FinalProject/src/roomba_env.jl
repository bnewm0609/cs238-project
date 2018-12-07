# Defines the environment as a POMDPs.jl MDP and POMDP
# maintained by {jmorton2,kmenda}@stanford.edu

known_commands = false
learn_directions = true

# Wraps ang to be in (-pi, pi]
function wrap_to_pi(ang::Float64)
    if ang > pi
		ang -= 2*pi
	elseif ang <= -pi
		ang += 2*pi
    end
	ang
end

"""
State of a Roomba.

# Fields
- `x::Float64` x location in meters
- `y::Float64` y location in meters
- `status::Float64` indicator whether robot has reached goal state
- `cmd_1::Float64`: mapping for command 1 in radians
- `cmd_2::Float64`: mapping for command 2 in radians
- `cmd_3::Float64`: mapping for command 3 in radians
- `cmd_4::Float64`: mapping for command 4 in radians
"""
@with_kw struct RoombaState <: FieldVector{7, Float64}
    x::Float64
    y::Float64
	status::Float64
	cmd_1::Float64 = pi # angles are (-pi to pi]
	cmd_2::Float64 = -pi/2.
	cmd_3::Float64 = 0.
	cmd_4::Float64 = pi/2.
end

# Struct for a Roomba action
struct RoombaAct <: FieldVector{1, Float64}
    theta::Float64     # direction of movement
	learn::Bool
end

# action spaces
struct RoombaActions end

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
- `sspace::SS` environment state-space (ContinuousRooxmbaStateSpace or DiscreteRoombaStateSpace)
- `aspace::AS` environment action-space struct
"""
@with_kw mutable struct RoombaMDP{SS,AS} <: MDP{RoombaState, RoombaAct}
    v::Float64  = 3  # m/s
    dt::Float64     = 0.5   # s
    contact_pen::Float64 = -1.0
    time_pen::Float64 = -0.1
    goal_reward::Float64 = 100
    room::Room  = Room()
    sspace::SS =  ContinuousRoombaStateSpace() # DiscreteRoombaStateSpace(30,30,24)
    aspace::AS = [RoombaAct(a, b) for b in [true] for a in range(-pi, length=25, stop=pi)[1:24]] # RoombaActions()
    _amap::Union{Nothing, Dict{RoombaAct, Int}} = gen_amap(aspace)
end

# state-space definitions
struct ContinuousRoombaStateSpace end

"""
Specify a DiscreteRoombaStateSpace
- `x_step::Float64` distance between discretized points in x
- `y_step::Float64` distance between discretized points in y
- `cmd_step::Float64` distance between discretized points in command mappings # For Modified Problem
- `XLIMS::Vector` boundaries of room (x-dimension)
- `YLIMS::Vector` boundaries of room (y-dimension)

"""
struct DiscreteRoombaStateSpace
    x_step::Float64
    y_step::Float64
    cmd_step::Float64  # For Modified Problem
    XLIMS::Vector
    YLIMS::Vector
end

# function to construct DiscreteRoombaStateSpace:
# `num_x_pts::Int` number of points to discretize x range to
# `num_y_pts::Int` number of points to discretize y range to
# `num_cmd_pts::Int` number of points to discretize command mapping range to # For Modified Problem
# function DiscreteRoombaStateSpace(num_x_pts::Int, num_y_pts::Int)
function DiscreteRoombaStateSpace(num_x_pts::Int, num_y_pts::Int, num_cmd_pts::Int) # For Modified Problem

    # hardcoded room-limits
    # watch for consistency with env_room
    XLIMS = [-30.0, 20.0]
    YLIMS = [-30.0, 10.0]

    return DiscreteRoombaStateSpace((XLIMS[2]-XLIMS[1])/(num_x_pts-1),
                                    (YLIMS[2]-YLIMS[1])/(num_y_pts-1),
                                    2*pi/(num_cmd_pts-1),  # For Modified Problem
                                    XLIMS,YLIMS)
end




"""
Define the Roomba POMDP

Fields:
- `sensor::T` struct specifying the sensor used (Lidar or Bump)
- `mdp::T` underlying RoombaMDP
"""
struct RoombaPOMDP{T, O} <: POMDP{RoombaState, RoombaAct, O}
    sensor::T
    mdp::RoombaMDP
end

# observation models
struct Bumper end
POMDPs.obstype(::Type{Bumper}) = Bool
POMDPs.obstype(::Bumper) = Bool

struct Lidar
    ray_stdev::Float64 # measurement noise: see POMDPs.observation definition
                       # below for usage
end
Lidar() = Lidar(0.1)

POMDPs.obstype(::Type{Lidar}) = Float64
POMDPs.obstype(::Lidar) = Float64 #float64(x)

struct DiscreteLidar
    ray_stdev::Float64
    disc_points::Vector{Float64} # cutpoints: endpoints of (0, Inf) assumed
end

POMDPs.obstype(::Type{DiscreteLidar}) = Int
POMDPs.obstype(::DiscreteLidar) = Int
DiscreteLidar(disc_points) = DiscreteLidar(Lidar().ray_stdev, disc_points)

struct Command
	dirs::Array{Tuple{Int64,Tuple{Float64,Float64}},1}
end
# commands are not noisy - they are completely deterministic
# the first value is the action, the second is the tuple (upper bound, lower bound)
# in the observation function the calculated theta has to be less than upper bound
# and greater than or equal to lower bound
# Note: the bounds for 1 might look reversed, but that's to handle wrapping. It
# The case is handled correctly in the POMDPs.observation function
Command() = Command([(1, (-3pi/4, 3pi/4)), (2, (-pi/4, -3pi/4)),
					(3, (pi/4, -pi/4)), (4, (3pi/4, pi/4))])
# Command() = Command([(1, (-pi/2, pi/2)), (2, (0, -pi)),
# 					(3, (pi/2, -pi/2)), (4, (pi, 0))])



POMDPs.obstype(::Type{Command}) = Int # 1, 2, 3, 4 for left, down, right, up
POMDPs.obstype(::Command) = Int



# Shorthands
const RoombaModel = Union{RoombaMDP, RoombaPOMDP}
const BumperPOMDP = RoombaPOMDP{Bumper, Bool}
const LidarPOMDP = RoombaPOMDP{Lidar, Float64}
const DiscreteLidarPOMDP = RoombaPOMDP{DiscreteLidar, Int}
const CommandPOMDP = RoombaPOMDP{Command, Int}

# access the mdp of a RoombaModel
mdp(e::RoombaMDP) = e
mdp(e::RoombaPOMDP) = e.mdp


# RoombaPOMDP Constructor
function RoombaPOMDP(sensor, mdp)
    RoombaPOMDP{typeof(sensor), obstype(sensor)}(sensor, mdp)
end

RoombaPOMDP(;sensor=Bumper(), mdp=RoombaMDP()) = RoombaPOMDP(sensor,mdp)

# function to determine if there is contact with a wall
wall_contact(e::RoombaModel, state) = wall_contact(mdp(e).room, state[1:2])

POMDPs.actions(m::RoombaModel) = mdp(m).aspace
POMDPs.n_actions(m::RoombaModel) = length(mdp(m).aspace) #TODO
# POMDPs.actions(m::RoombaModel) = [RoombaAct(a) for a in range(-pi, length=25, stop=pi)[1:24]]
# POMDPs.n_actions(m::RoombaModel) = 24

# maps a RoombaAct to an index in a RoombaModel with discrete actions
function POMDPs.actionindex(m::RoombaModel, a::RoombaAct)
    if mdp(m)._amap != nothing
        return mdp(m)._amap[a]
    else
        error("Action index not defined for continuous actions.")
    end
end

# function to get goal xy location for heuristic controllers
function get_goal_pos(m::RoombaModel)
    return mdp(m).room.goal_pos
end

function at_goal(cmd_1, cmd_2, cmd_3, cmd_4, x, y, m::RoombaModel)
    goal_x, goal_y = get_goal_pos(m)
	# return (abs(x - goal_x) < 2) && abs(y - goal_y) < 2 # TODO: Make a more robust buffer
	d(x,y) = abs(atan(sin(x-y), cos(x-y)))/pi
	angle_diff_sum = d(cmd_1, pi) + d(cmd_2, -pi/2) + d(cmd_3, 0.) + d(cmd_4, pi/2.)
	if learn_directions
		return (abs(x - goal_x) < 2) && abs(y - goal_y) < 2 && d(cmd_1, pi) < 0.25 && d(cmd_2, -pi/2) < 0.25 && d(cmd_3, 0.) < 0.25 && d(cmd_4, pi/2.) < 0.25
	else
		return (abs(x - goal_x) < 2) && abs(y - goal_y) < 2
	end
end

# initializes x,y of Roomba in the room
function POMDPs.initialstate(m::RoombaModel, rng::AbstractRNG)
    e = mdp(m)
    x, y = init_pos(e.room, rng)
	status = 0.0
	cmd_1 = known_commands ? pi     : rand() * 2*pi - pi
	cmd_2 = known_commands ? -pi/2. : rand() * 2*pi - pi
	cmd_3 = known_commands ? 0. 	: rand() * 2*pi - pi
	cmd_4 = known_commands ? pi/2.	: rand() * 2*pi - pi

	# TODO: confirm that we don't need to do this
	# while(at_goal(x,y,m))
	# 	println("Goal pos: $(get_goal_pos(m)); Init pos: ($x, $y)")
	# 	x, y = init_pos(e.room, rng)
	# end

	is = RoombaState(x=x, y=y, status=status, cmd_1=cmd_1, cmd_2=cmd_2, cmd_3=cmd_3, cmd_4=cmd_4)

    if mdp(m).sspace isa DiscreteRoombaStateSpace
        isi = stateindex(m, is)
        is = index_to_state(m, isi)
    end

    return is
end

# transition Roomba state given curent state and action
function POMDPs.transition(m::RoombaModel,
                           s::AbstractVector{Float64},
                           a)#::AbstractVector{Float64})

    e = mdp(m)
    theta = a[1][1][1]
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

    # define next state

	d(x,y) = abs(atan(sin(x-y), cos(x-y)))/pi
	closest = argmin([d(theta, cmd) for cmd in [s.cmd_1, s.cmd_2, s.cmd_3, s.cmd_4]])
	new_cmd_1 = s.cmd_1
	new_cmd_2 = s.cmd_2
	new_cmd_3 = s.cmd_3
	new_cmd_4 = s.cmd_4

	if !known_commands && a[2] == true
		η = 0.3

		cmd_received = speaker_response(m, s).val
		@assert cmd_received == 1 || cmd_received == 2 || cmd_received == 3 || cmd_received == 4

		opp = wrap_to_pi(pi + theta)

		# new_cmd_1 += η*(opp - s.cmd_1)
		# new_cmd_2 += η*(opp - s.cmd_2)
		# new_cmd_3 += η*(opp - s.cmd_3)
		# new_cmd_4 += η*(opp - s.cmd_4)

		if cmd_received == 1
			# new_cmd_1 -= η*(opp - s.cmd_1)
			new_cmd_1 += η*(theta - s.cmd_1)
		elseif cmd_received == 2
			# new_cmd_2 -= η*(opp - s.cmd_2)
			new_cmd_2 += η*(theta - s.cmd_2)
		elseif cmd_received == 3
			# new_cmd_3 -= η*(opp - s.cmd_3)
			new_cmd_3 += η*(theta - s.cmd_3)
		elseif cmd_received == 4
			# new_cmd_4 -= η*(opp - s.cmd_4)
			new_cmd_4 += η*(theta - s.cmd_4)
		end

	end

	# Determine whether goal has been reached
    next_status = 1.0*at_goal(new_cmd_1, new_cmd_2, new_cmd_3, new_cmd_4, next_x, next_y, m)

    sp = RoombaState(x=next_x, y=next_y, status=next_status, cmd_1=new_cmd_1, cmd_2=new_cmd_2, cmd_3=new_cmd_3, cmd_4=new_cmd_4)

    if mdp(m).sspace isa DiscreteRoombaStateSpace
        # round the states to nearest grid point
        si = stateindex(m, sp)
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
		statuses = [0.,1.]
        cmd_1_states = known_commands ? [pi] : range(-pi, stop=pi, step=ss.cmd_step)  # For Modified Problem
		cmd_2_states = known_commands ? [-pi/2.] : range(-pi, stop=pi, step=ss.cmd_step)
		cmd_3_states = known_commands ? [0.] : range(-pi, stop=pi, step=ss.cmd_step)
		cmd_4_states = known_commands ? [pi/2.] : range(-pi, stop=pi, step=ss.cmd_step)
        return vec(collect(RoombaState(x=x,y=y,cmd_1=cmd_1,cmd_2=cmd_2,cmd_3=cmd_3,cmd_4=cmd_4,status=st)
			for x in x_states, y in y_states, st in statuses, cmd_1 in cmd_1_states, cmd_2 in cmd_2_states, cmd_3 in cmd_3_states, cmd_4 in cmd_4_states))
    else
        return mdp(m).sspace
    end
end

# return the number of states in a DiscreteRoombaStateSpace
function POMDPs.n_states(m::RoombaModel)
    if mdp(m).sspace isa DiscreteRoombaStateSpace
        ss = mdp(m).sspace
    		nstates = known_commands ? prod((convert(Int, diff(ss.XLIMS)[1]/ss.x_step)+1,
                            convert(Int, diff(ss.YLIMS)[1]/ss.y_step)+1,
                            2)) : prod((convert(Int, diff(ss.XLIMS)[1]/ss.x_step)+1,  # For Modified Problem
                            convert(Int, diff(ss.YLIMS)[1]/ss.y_step)+1,
							2,
                            round(Int, 2*pi/ss.cmd_step)+1,
  							round(Int, 2*pi/ss.cmd_step)+1,
			                round(Int, 2*pi/ss.cmd_step)+1,
			              	round(Int, 2*pi/ss.cmd_step)+1))
        return nstates
    else
        error("State-space must be DiscreteRoombaStateSpace.")
    end
end

# map a RoombaState to an index in a DiscreteRoombaStateSpace
function POMDPs.stateindex(m::RoombaModel, s::RoombaState)
    if mdp(m).sspace isa DiscreteRoombaStateSpace
        ss = mdp(m).sspace
        xind = floor(Int, (s.x - ss.XLIMS[1]) / ss.x_step + 0.5) + 1
        yind = floor(Int, (s.y - ss.YLIMS[1]) / ss.y_step + 0.5) + 1
		stind = convert(Int, s.status + 1)
		cmd1ind = known_commands ? 1 : floor(Int, (s.cmd_1 - (-pi)) / ss.cmd_step + 0.5) + 1  # For Modified Problem
		cmd2ind = known_commands ? 1 : floor(Int, (s.cmd_2 - (-pi)) / ss.cmd_step + 0.5) + 1
		cmd3ind = known_commands ? 1 : floor(Int, (s.cmd_3 - (-pi)) / ss.cmd_step + 0.5) + 1
		cmd4ind = known_commands ? 1 : floor(Int, (s.cmd_4 - (-pi)) / ss.cmd_step + 0.5) + 1

        lin = known_commands ? LinearIndices((convert(Int, diff(ss.XLIMS)[1]/ss.x_step)+1,
                            convert(Int, diff(ss.YLIMS)[1]/ss.y_step)+1,
							2,
							1, 1, 1, 1
							)) : LinearIndices((convert(Int, diff(ss.XLIMS)[1]/ss.x_step)+1,  # For Modified Problem
                     		convert(Int, diff(ss.YLIMS)[1]/ss.y_step)+1,
							2,
	                        round(Int, 2*pi/ss.cmd_step)+1,
		                    round(Int, 2*pi/ss.cmd_step)+1,
		                    round(Int, 2*pi/ss.cmd_step)+1,
	                     	round(Int, 2*pi/ss.cmd_step)+1
							))
        return lin[xind,yind,stind,cmd1ind,cmd2ind,cmd3ind,cmd4ind]  # For Modified Problem
    else
        error("State-space must be DiscreteRoombaStateSpace.")
    end
end

# map an index in a DiscreteRoombaStateSpace to the corresponding RoombaState
function index_to_state(m::RoombaModel, si::Int)
    if mdp(m).sspace isa DiscreteRoombaStateSpace
        ss = mdp(m).sspace
        lin = known_commands ? CartesianIndices((convert(Int, diff(ss.XLIMS)[1]/ss.x_step)+1,
                            convert(Int, diff(ss.YLIMS)[1]/ss.y_step)+1,
                            2)) : CartesianIndices((convert(Int, diff(ss.XLIMS)[1]/ss.x_step)+1,  # For Modified Problem
		                    convert(Int, diff(ss.YLIMS)[1]/ss.y_step)+1,
	                      	2,
							round(Int, 2*pi/ss.cmd_step)+1,
		                    round(Int, 2*pi/ss.cmd_step)+1,
		                    round(Int, 2*pi/ss.cmd_step)+1,
		                    round(Int, 2*pi/ss.cmd_step)+1))

		xi,yi,sti,cmd1i,cmd2i,cmd3i,cmd4i = Tuple(lin[si])  # For Modified Problem

        x = ss.XLIMS[1] + (xi-1) * ss.x_step
        y = ss.YLIMS[1] + (yi-1) * ss.y_step
        st = sti - 1
		cmd_1 = known_commands ? pi    : -pi + (cmd1i-1) * ss.cmd_step
		cmd_2 = known_commands ? -pi/2. : -pi + (cmd2i-1) * ss.cmd_step
		cmd_3 = known_commands ? 0.     : -pi + (cmd3i-1) * ss.cmd_step
		cmd_4 = known_commands ? pi/2.  : -pi + (cmd4i-1) * ss.cmd_step

        return RoombaState(x=x, y=y, status=st, cmd_1=cmd_1, cmd_2=cmd_2, cmd_3=cmd_3, cmd_4=cmd_4)

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

	# abs_dot(x, y) = abs(sin(x) * sin(y) + cos(x) * cos(y))
	# mapping_cost = 0
	# mapping_cost += abs_dot(s.cmd_1, s.cmd_2)
	# mapping_cost += abs_dot(s.cmd_2, s.cmd_3)
	# mapping_cost += abs_dot(s.cmd_3, s.cmd_4)
	# mapping_cost += abs_dot(s.cmd_4, s.cmd_1)
	# cum_reward -= 10 * mapping_cost

    return cum_reward
end

# determine if a terminal state has been reached
POMDPs.isterminal(m::RoombaModel, s::AbstractVector{Float64}) = abs(s.status) > 0.0

# Bumper POMDP observation
function POMDPs.observation(m::BumperPOMDP,
                            a::AbstractVector{Float64},
                            sp::AbstractVector{Float64})
    return Deterministic(wall_contact(m, sp)) # in {0.0,1.0}
end

POMDPs.n_observations(m::BumperPOMDP) = 2
POMDPs.observations(m::BumperPOMDP) = [false, true]

# Lidar POMDP observation
function POMDPs.observation(m::LidarPOMDP,
                            a::AbstractVector{Float64},
                            sp::AbstractVector{Float64})
    x, y, th = sp

    # determine uncorrupted observation
    rl = ray_length(mdp(m).room, [x, y], [cos(th), sin(th)])

    # compute observation noise
    sigma = m.sensor.ray_stdev * max(rl, 0.01)

    # disallow negative measurements
    return Truncated(Normal(rl, sigma), 0.0, Inf)
end

function POMDPs.n_observations(m::LidarPOMDP)
    error("n_observations not defined for continuous observations.")
end

function POMDPs.observations(m::LidarPOMDP)
    error("LidarPOMDP has continuous observations. Use DiscreteLidarPOMDP for discrete observation spaces.")
end

# DiscreteLidar POMDP observation
function POMDPs.observation(m::DiscreteLidarPOMDP,
                            a::AbstractVector{Float64},
                            sp::AbstractVector{Float64})

    m_lidar = LidarPOMDP(Lidar(m.sensor.ray_stdev), mdp(m))

    d = observation(m_lidar, a, sp)

    disc_points = [-Inf, m.sensor.disc_points..., Inf]

    d_disc = diff(cdf.(d, disc_points))

    return SparseCat(1:length(d_disc), d_disc)
end

POMDPs.n_observations(m::DiscreteLidarPOMDP) = length(m.sensor.disc_points) + 1
POMDPs.observations(m::DiscreteLidarPOMDP) = vec(1:n_observations(m))

function speaker_response(m::CommandPOMDP,
				   sp::AbstractVector{Float64})
	# basically what we have to do is get the direction to the room and then
	# use some tie-breaking scheme to decide what to return
	x, y = sp # assume that the first two elements of the state are x, y
			  # (In Julia, you only have to unpack the first n elements of tuples)
	gx, gy = get_goal_pos(m)

	# calculate angle to goal using arctan - returns an angle between -pi and pi
	# Awesome note: in Julia you can type "pi" and get the value of pi....
	th_goal = atan(gy - y, gx - x)

	# choose action - note that there is a very explicit tie-break assumption
	# here - we are choosing randomly
	dirs = Random.shuffle(m.sensor.dirs)

	for (cmd, dir) in dirs
		if cmd == 1 # because of wrapping, we need to handle this case differently
			if th_goal <= dir[1] || th_goal > dir[2]
				return Deterministic(cmd)
			end
		elseif th_goal <= dir[1] && th_goal > dir[2]
			return Deterministic(cmd)
		end
	end
	@assert false # we shouldn't get here
end

# new stuff
function POMDPs.observation(m::CommandPOMDP,
				   a::AbstractVector{Float64},
				   sp::AbstractVector{Float64})
	return speaker_response(m, sp)
end

POMDPs.n_observations(m::CommandPOMDP) = length(m.sensor.dirs)
POMDPs.observations(m::CommandPOMDP) = vec(1:4)

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
				render_command_mapping(ctx, p, true)
            end
        end
    end

    # Render room
    render(env.room, ctx)

	# Render commands
	render_commands(ctx)

    # Find center of robot in frame and draw circle
    x, y = transform_coords(state[1:2])
    arc(ctx, x, y, radius, 0, 2*pi)
    set_source_rgb(ctx, 1, 0.6, 0.6)
    fill(ctx)

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

	render_command_mapping(ctx, state, false)

    return ctx
end

function render_command_mapping(ctx::CairoContext, state, belief::Bool)
	radius = ROBOT_W*6

	ctr_x = 120
	ctr_y = 500
	for i = 4:7
		x = ctr_x * (i-3) + 50 * cos(state[i])
		y = ctr_y - 50 * sin(state[i]) # TODO: check; because y coordinates are flipped
		arc(ctx, x, y, radius, 0, 2*pi)
		if belief
			set_source_rgba(ctx, 0.6, 0.6, 1, 0.3)
		else
			set_source_rgba(ctx, 1.0, 0.6, 0.6, 1.0)
		end
		fill(ctx)
	end
end

function render_commands(ctx::CairoContext)
	for i = 1:4
		arc(ctx, i*(600-2*60)/4, 500, 60, 0, 2*pi)
		set_source_rgba(ctx, 0.8, 0.8, 0.8, 0.3)
		fill(ctx)
	end
end

function render(m::RoombaModel, step)
    io = IOBuffer()
    c = CairoSVGSurface(io, 800, 600)
    ctx = CairoContext(c)
    render(ctx, m, step)
    finish(c)
    return HTML(String(take!(io)))
end
