"""
$(TYPEDEF)

Variational optimization algorithm for excitations of finite MPS by minimizing the energy of

```math
H - λᵢ |ψᵢ⟩⟨ψᵢ|
```

## Fields

$(TYPEDFIELDS)
"""
@kwdef struct FiniteExcited{A} <: Algorithm
    "optimization algorithm"
    gsalg::A = DMRG()
    "energy penalty for enforcing orthogonality with previous states"
    weight::Float64 = 10.0
end

function excitations(H::FiniteMPOHamiltonian, alg::FiniteExcited,
                     states::Tuple{T,Vararg{T}};
                     init=FiniteMPS([copy(first(states).AC[i])
                                     for i in 1:length(first(states))]),
                     num=1,
                     which=:SR,) where {T<:FiniteMPS}
    num == 0 && return (scalartype(T)[], T[])
    energy = 
    if which==:LM
        super_op = LinearCombination(tuple(H, ProjectionOperator.(states)...),
                                 tuple(1.0, broadcast(x -> -0.1*alg.weight, states)...))
    else
        super_op = LinearCombination(tuple(H, ProjectionOperator.(states)...),
                                 tuple(1.0, broadcast(x -> alg.weight, states)...))
    end
    envs = environments(init, super_op)
    ne, _ = find_groundstate(init, super_op, alg.gsalg, envs; which=which)

    nstates = (states..., ne)
    ens, excis = excitations(H, alg, nstates; init=init, num=num - 1, which=which)

    push!(ens, expectation_value(ne, H))
    push!(excis, ne)

    return ens, excis
end

function excitations(H::FiniteMPO, alg::FiniteExcited,
                     states::Tuple{T,Vararg{T}};
                     init=FiniteMPS([copy(first(states).AC[i])
                                     for i in 1:length(first(states))]),
                     num=1,
                     which=:SR,) where {T<:FiniteMPS}
    num == 0 && return (scalartype(T)[], T[])
    if which==:LM
        energies = [expectation_value(psi,H) for psi in states]
        super_op = LinearCombination(tuple(H, ProjectionOperator.(states)...),
                                 tuple(1.0, -energies...))
    else
        super_op = LinearCombination(tuple(H, ProjectionOperator.(states)...),
                                 tuple(1.0, broadcast(x -> alg.weight, states)...))
    end
    envs = environments(init, super_op)
    ne, _ = find_groundstate(init, super_op, alg.gsalg, envs; which=which)

    nstates = (states..., ne)
    ens, excis = excitations(H, alg, nstates; init=init, num=num - 1, which=which)

    push!(ens, expectation_value(ne, H))
    push!(excis, ne)

    return ens, excis
end

function excitations(H, alg::FiniteExcited, ψ::FiniteMPS; kwargs...)
    return excitations(H, alg, (ψ,); kwargs...)
end
