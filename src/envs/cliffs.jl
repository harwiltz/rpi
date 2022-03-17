module CliffsOfMoherModule

##### See https://www.distributional-rl.org/contents/chapter2.html

import GridWorlds as GW
import Random
import ReinforcementLearningBase as RLBase

const NUM_OBJECTS = 2
const AGENT = 1
const CLIFF = 2
const NUM_ACTIONS = 4

const SLIP_PROB = 0.25
const WIDTH = 12
const HEIGHT = 4

const INITIAL_MAP = let
    tile_map = falses(NUM_OBJECTS, HEIGHT, WIDTH)
    tile_map[CLIFF, end, 2:end - 1] .= true
    tile_map[AGENT, end, 1] = true
    tile_map
end

const CHARACTERS = ('☻', '█', '◎', '.')

mutable struct CliffsOfMoher{R, RNG} <: GW.AbstractGridWorld
    tile_map::BitArray{3} # Object type, x, y
    agent_position::CartesianIndex{2}
    reward::R
    rng::RNG
    done::Bool
    terminal_reward::R
    terminal_penalty::R
    target_position::CartesianIndex{2}
end

function CliffsOfMoher(; R = Float32, rng = Random.GLOBAL_RNG)
    reward = zero(R)
    terminal_reward = one(R)
    terminal_penalty = -one(R)
    env = CliffsOfMoher(copy(INITIAL_MAP),
                        CartesianIndex(1, 1),
                        reward,
                        rng,
                        false,
                        terminal_reward,
                        terminal_penalty,
                        CartesianIndex(HEIGHT, WIDTH))
    GW.reset!(env)
    return env
end

function GW.reset!(env::CliffsOfMoher)
    env.tile_map = copy(INITIAL_MAP)
    env.agent_position = CartesianIndex(HEIGHT, 1)
    env.done = false
    env.reward = zero(env.reward)
end

function GW.act!(env::CliffsOfMoher, action)
    @assert action in Base.OneTo(NUM_ACTIONS) "Invalid action $(action). Action must be in Base.OneTo($(NUM_ACTIONS))."

    tile_map = env.tile_map
    rng = env.rng
    pos = env.agent_position

    if rand(rng) <= SLIP_PROB
        action = rand(rng, Base.OneTo(NUM_ACTIONS))
    end
    
    if action == 1
        pos′ = move_up_bounded(pos, WIDTH, HEIGHT)
    elseif action == 2
        pos′ = move_down_bounded(pos, WIDTH, HEIGHT)
    elseif action == 3
        pos′ = move_left_bounded(pos, WIDTH, HEIGHT)
    else
        pos′ = move_right_bounded(pos, WIDTH, HEIGHT)
    end

    if tile_map[CLIFF, pos′]
        env.done = true
        env.reward = env.terminal_penalty
    elseif pos′ == env.target_position
        env.done = true
        env.reward = env.terminal_reward
    else
        env.tile_map[AGENT, pos] = false
        env.tile_map[AGENT, pos′] = true
        env.agent_position = pos′
        env.reward = zero(env.reward)
    end
end

move_up_bounded(pos::CartesianIndex{2}, width, height) = CartesianIndex(max(pos[1] - 1, 1), pos[2])
move_down_bounded(pos::CartesianIndex{2}, width, height) = CartesianIndex(min(pos[1] + 1, height), pos[2])
move_left_bounded(pos::CartesianIndex{2}, width, height) = CartesianIndex(pos[1], max(1, pos[2] - 1))
move_right_bounded(pos::CartesianIndex{2}, width, height) = CartesianIndex(pos[1], min(width, pos[2] + 1))

GW.get_height(env::CliffsOfMoher) = size(env.tile_map, 2)
GW.get_width(env::CliffsOfMoher) = size(env.tile_map, 3)
GW.get_action_names(env::CliffsOfMoher) = (:MOVE_UP, :MOVE_DOWN, :MOVE_LEFT, :MOVE_RIGHT)
GW.get_object_names(env::CliffsOfMoher) = (:AGENT, :CLIFF)

function GW.get_pretty_tile_map(env::CliffsOfMoher, position::CartesianIndex{2})
    if position == env.target_position
        return CHARACTERS[end - 1]
    end
    
    object = findfirst(@view env.tile_map[:, position])
    if isnothing(object)
        CHARACTERS[end]
    else
        CHARACTERS[object]
    end
end

function GW.get_pretty_sub_tile_map(env::CliffsOfMoher, window_size, position::CartesianIndex{2})
    tile_map = env.tile_map
    agent_position = env.agent_position
    sub_tile_map = GW.get_sub_tile_map(tile_map, agent_position, window_size)
    if position == env.target_position
        return CHARACTERS[end - 1]
    end

    object = findfirst(@view sub_tile_map[:, position])
    if isnothing(object)
        CHARACTERS[end]
    else
        CHARACTERS[object]
    end
end

function Base.show(io::IO, ::MIME"text/plain", env::CliffsOfMoher)
    str = "tile_map:\n"
    str = str * GW.get_pretty_tile_map(env)
    str = str * "\nsub_tile_map:\n"
    str = str * GW.get_pretty_sub_tile_map(env, GW.get_window_size(env))
    str = str * "\nreward: $(env.reward)"
    str = str * "\ndone: $(env.done)"
    str = str * "\nagent_position: $(env.agent_position)"
    str = str * "\naction_names: $(GW.get_action_names(env))"
    str = str * "\nobject_names: $(GW.get_object_names(env))\n"
    print(io, str)
end

GW.get_action_keys(env::CliffsOfMoher) = ('k', 'j', 'h', 'l')

RLBase.StateStyle(env::GW.RLBaseEnv{E}) where {E <: CliffsOfMoher} = RLBase.Observation{Any}()
RLBase.state_space(env::GW.RLBaseEnv{E}, ::RLBase.Observation) where {E <: CliffsOfMoher} = nothing
RLBase.state(env::GW.RLBaseEnv{E}, ::RLBase.Observation) where {E <: CliffsOfMoher} = env.env.tile_map
RLBase.reset!(env::GW.RLBaseEnv{E}) where {E <: CliffsOfMoher} = GW.reset!(env.env)
RLBase.action_space(env::GW.RLBaseEnv{E}) where {E <: CliffsOfMoher} = Base.OneTo(NUM_ACTIONS)
(env::GW.RLBaseEnv{E})(action) where {E <: CliffsOfMoher} = GW.act!(env.env, action)
RLBase.reward(env::GW.RLBaseEnv{E}) where {E <: CliffsOfMoher} = env.env.reward
RLBase.is_terminated(env::GW.RLBaseEnv{E}) where {E <: CliffsOfMoher} = env.env.done

end # module
