export BaseRiskSensitiveLearner

mutable struct BaseRiskSensitiveLearner{A, B₀, B} <: AbstractLearner
    approximator::A
    γ::Float64
    b::B₀
    ℬ::B
    τ::Float32
end

function BaseRiskSensitiveLearner(approximator, ℬ, τ = 0.5; γ = 0.99, B = Float32, b = 0)
    BaseRiskSensitiveLearner(approximator, γ, b, ℬ, τ)
end

function RLBase.update!(
    t::AbstractTrajectory,
    learner::BaseRiskSensitiveLearner,
    env::AbstractEnv,
    ::PreEpisodeStage
)
    empty!(t)
    X₀ = state(env)
    learner.b = initial_b(learner, X₀)
end

function initial_b(
    learner::BaseRiskSensitiveLearner,
    X₀
)
    ℬ = learner.ℬ
