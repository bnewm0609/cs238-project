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

"""
State of a Roomba.

# Fields
- `x::Float64` x location in meters
- `y::Float64` y location in meters
- `cmd_1::Float64`: mapping for command 1 in radians
- `cmd_2::Float64`: mapping for command 2 in radians
- `cmd_3::Float64`: mapping for command 3 in radians
- `cmd_4::Float64`: mapping for command 4 in radians
- `status::Float64` indicator whether robot has reached goal state
"""
@with_kw struct RoombaState <: FieldVector{7, Float64}
    x::Float64
    y::Float64
	status::Float64
	cmd_1::Float64 = pi # angles are (-pi to pi]
	cmd_2::Float64 = -pi/2
	cmd_3::Float64 = 0.
	cmd_4::Float64 = pi/2
end

# Struct for a Roomba action
struct RoombaAct <: FieldVector{1, Float64}
    theta::Float64     # direction of movement
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
- `sspace::SS` environment state-space (ContinuousRoombaStateSpace or DiscreteRoombaStateSpace)
- `aspace::AS` environment action-space struct
"""
@with_kw mutable struct RoombaMDP{SS,AS} <: MDP{RoombaState, RoombaAct}
    v::Float64  = 0.5  # m/s
    dt::Float64     = 0.5   # s
    contact_pen::Float64 = -1.0
    time_pen::Float64 = -0.1
    goal_reward::Float64 = 10
    room::Room  = Room()
    sspace::SS = ContinuousRoombaStateSpace()
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
    # cmd_step::Float64  # For Modified Problem
    XLIMS::Vector
    YLIMS::Vector
end

# function to construct DiscreteRoombaStateSpace:
# `num_x_pts::Int` number of points to discretize x range to
# `num_y_pts::Int` number of points to discretize y range to
## `num_cmd_pts::Int` number of points to discretize command mapping range to # For Modified Problem
function DiscreteRoombaStateSpace(num_x_pts::Int, num_y_pts::Int)
# function DiscreteRoombaStateSpace(num_x_pts::Int, num_y_pts::Int, num_cmd_pts::Int) # For Modified Problem

    # hardcoded room-limits
    # watch for consistency with env_room
    XLIMS = [-30.0, 20.0]
    YLIMS = [-30.0, 10.0]

    return DiscreteRoombaStateSpace((XLIMS[2]-XLIMS[1])/(num_x_pts-1),
                                    (YLIMS[2]-YLIMS[1])/(num_y_pts-1),
#                                    2*pi/(num_cmd_pts-1),  # For Modified Problem
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
POMDPs.n_actions(m::RoombaModel) = length(mdp(m).aspace)

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

function at_goal(x, y, m::RoombaModel)
    goal_x, goal_y = get_goal_pos(m)
	return (abs(x - goal_x) < .2) && abs(y - goal_y) < .2 # TODO: Make a more robust buffer
end

# initializes x,y of Roomba in the room
function POMDPs.initialstate(m::RoombaModel, rng::AbstractRNG)
    e = mdp(m)
    x, y = init_pos(e.room, rng)
	# cmd_1 = rand() * 2*pi - pi
	# cmd_2 = rand() * 2*pi - pi
	# cmd_3 = rand() * 2*pi - pi
	# cmd_4 = rand() * 2*pi - pi

	while(at_goal(x,y,m))
		println("Goal pos: $(get_goal_pos(m)); Init pos: ($x, $y)")
		x, y = init_pos(e.room, rng)
	end

    is = RoombaState(x=x, y=y, status=0.0)
	# is = RoombaState(x=x, y=y, cmd_1=cmd_1, cmd_2=cmd_2, cmd_3=cmd_3, cmd_4=cmd_4 status=0.0)

    if mdp(m).sspace isa DiscreteRoombaStateSpace
        isi = stateindex(m, is)
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

    # define next state
    sp = RoombaState(x=next_x, y=next_y, status=next_status)

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
        # cmd_states = range(-pi, stop=pi, step=ss.cmd_step)  # For Modified Problem
        statuses = [0.,1.]
        return vec(collect(RoombaState(x=x,y=y,status=st) for x in x_states, y in y_states, st in statuses))
        # return vec(collect(RoombaState(x=x,y=y,cmd_1=cmd_1,cmd_2=cmd_2,cmd_3=cmd_3,cmd_4=cmd_4,status=st) for x in x_states, y in y_states, cmd_1 in cmd_states, cmd_2 in cmd_states, cmd_3 in cmd_states, cmd_4 in cmd_states, st in statuses))
    else
        return mdp(m).sspace
    end
end

# return the number of states in a DiscreteRoombaStateSpace
function POMDPs.n_states(m::RoombaModel)
    if mdp(m).sspace isa DiscreteRoombaStateSpace
        ss = mdp(m).sspace
    		nstates = prod((convert(Int, diff(ss.XLIMS)[1]/ss.x_step)+1,
                            convert(Int, diff(ss.YLIMS)[1]/ss.y_step)+1,
                            2))
        # nstates = prod((convert(Int, diff(ss.XLIMS)[1]/ss.x_step)+1,  # For Modified Problem
                            # convert(Int, diff(ss.YLIMS)[1]/ss.y_step)+1,
                            # round(Int, 2*pi/ss.cmd_step)+1,
              							# round(Int, 2*pi/ss.cmd_step)+1,
							              # round(Int, 2*pi/ss.cmd_step)+1,
						              	# round(Int, 2*pi/ss.cmd_step)+1,
                            # 2))
        return nstates
    else
        error("State-space must be DiscreteRoombaStateSpace.")
    end
end

# map a RoombaState to an index in a DiscreteRoombaStateSpace
function POMDPs.stateindex(m::RoombaModel, s::RoombaState)
    if mdp(m).sspace isa DiscreteRoombaStateSpace
        ss = mdp(m).sspace
        xind = floor(Int, (s[1] - ss.XLIMS[1]) / ss.x_step + 0.5) + 1
        yind = floor(Int, (s[2] - ss.YLIMS[1]) / ss.y_step + 0.5) + 1
        stind = convert(Int, s[3] + 1)
		# cmd1ind = floor(Int, (s[4] - (-pi)) / ss.cmd_step + 0.5) + 1  # For Modified Problem
		# cmd2ind = floor(Int, (s[5] - (-pi)) / ss.cmd_step + 0.5) + 1
		# cmd3ind = floor(Int, (s[6] - (-pi)) / ss.cmd_step + 0.5) + 1
		# cmd4ind = floor(Int, (s[7] - (-pi)) / ss.cmd_step + 0.5) + 1

        lin = LinearIndices((convert(Int, diff(ss.XLIMS)[1]/ss.x_step)+1,
                            convert(Int, diff(ss.YLIMS)[1]/ss.y_step)+1,
                            2))
		# lin = LinearIndices((convert(Int, diff(ss.XLIMS)[1]/ss.x_step)+1,  # For Modified Problem
	    #                     convert(Int, diff(ss.YLIMS)[1]/ss.y_step)+1,
	    #                     round(Int, 2*pi/ss.cmd_step)+1,
		  #                     round(Int, 2*pi/ss.cmd_step)+1,
		  #                     round(Int, 2*pi/ss.cmd_step)+1,
		  #                     round(Int, 2*pi/ss.cmd_step)+1,
	    #                     2))
		return lin[xind,yind,stind]
        # return lin[xind,yind,stind,cmd1ind,cmd2ind,cmd3ind,cmd4ind]  # For Modified Problem
    else
        error("State-space must be DiscreteRoombaStateSpace.")
    end
end

# map an index in a DiscreteRoombaStateSpace to the corresponding RoombaState
function index_to_state(m::RoombaModel, si::Int)
    if mdp(m).sspace isa DiscreteRoombaStateSpace
        ss = mdp(m).sspace
        lin = CartesianIndices((convert(Int, diff(ss.XLIMS)[1]/ss.x_step)+1,
                            convert(Int, diff(ss.YLIMS)[1]/ss.y_step)+1,
                            2))
		# lin = CartesianIndices((convert(Int, diff(ss.XLIMS)[1]/ss.x_step)+1,  # For Modified Problem
		#                     convert(Int, diff(ss.YLIMS)[1]/ss.y_step)+1,
	  #                     round(Int, 2*pi/ss.cmd_step)+1,
		#                     round(Int, 2*pi/ss.cmd_step)+1,
		#                     round(Int, 2*pi/ss.cmd_step)+1,
		#                     round(Int, 2*pi/ss.cmd_step)+1,
	  #                     2))

        xi,yi,sti = Tuple(lin[si])
		# xi,yi,sti,cmd1i,cmd2i,cmd3i,cmd4i = Tuple(lin[si])  # For Modified Problem

        x = ss.XLIMS[1] + (xi-1) * ss.x_step
        y = ss.YLIMS[1] + (yi-1) * ss.y_step
        st = sti - 1
		# cmd_1 = -pi + (cmd1i-1) * ss.cmd_step
		# cmd_2 = -pi + (cmd2i-1) * ss.cmd_step
		# cmd_3 = -pi + (cmd3i-1) * ss.cmd_step
		# cmd_4 = -pi + (cmd4i-1) * ss.cmd_step

        return RoombaState(x=x, y=y, status=st)

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


# new stuff
function POMDPs.observation(m::CommandPOMDP,
				   a::AbstractVector{Float64},
				   sp::AbstractVector{Float64})
	# basically what we have to do is get the direction to the room and then
	# use some tie-breaking scheme to decide what to return
	x, y = sp # assume that the first two elements of the state are x, y
			  # (In Julia, you only have to unpack the first n elemenst of tuples)
	gx, gy = get_goal_pos(m)

	# calculate angle to goal using arctan - returns an angle between -pi and pi
	# Awesome note: in Julia you can type "pi" and get the value of pi....
	th_goal = atan(gx - x, gy - y)

	# choose action - note that there is a very explicit tie-break assumption
	# here - we are choosing randomly
	dirs = Random.shuffle(m.sensor.dirs)

	for (a, dir) in dirs
		if a == 1 # because of wrapping, we need to handle this case differently
			if th_goal <= dir[1] || th_goal > dir[2]
				return Deterministic(a)
			end
		elseif th_goal <= dir[1] && th_goal > dir[2]
			return Deterministic(a)
		end
	end
	@assert false # we shouldn't get here
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
function render(ctx::CairoContext, m::RoombaModel, step)
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
