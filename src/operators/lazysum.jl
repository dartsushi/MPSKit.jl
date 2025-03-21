"""
    LazySum{O} <: AbstractVector{O}

Type that represents a lazy sum i.e explicit summation is only done when needed. 
This type is basically an `AbstractVector` with some extra functionality to calculate things efficiently.

## Fields
- ops -- Vector of summable objects

---

## Constructors
    LazySum(x::Vector)

"""
struct LazySum{O} <: AbstractVector{O}
    ops::Vector{O}
end

# for the AbstractArray interface
Base.size(x::LazySum) = size(x.ops)
Base.getindex(x::LazySum, i::Int) = x.ops[i]
Base.length(x::LazySum) = prod(size(x))
Base.similar(x::LazySum, ::Type{S}, dims::Dims) where {S} = LazySum(similar(x.ops, S, dims))
Base.setindex!(A::LazySum, X, i::Int) = (setindex!(A.ops, X, i); A)

Base.complex(x::LazySum) = LazySum(complex.(x.ops))

# Holy traits
TimeDependence(x::LazySum) = istimed(x) ? TimeDependent() : NotTimeDependent()
istimed(x::LazySum) = any(istimed, x)

# constructors
LazySum(x) = LazySum([x])
LazySum(ops::AbstractVector, fs::AbstractVector) = LazySum(map(MultipliedOperator, ops, fs))

# wrapper around _eval_at
safe_eval(::TimeDependent, x::LazySum, t::Number) = map(O -> _eval_at(O, t), x)
function safe_eval(::TimeDependent, x::LazySum)
    throw(ArgumentError("attempting to evaluate time-dependent LazySum without specifiying a time"))
end
safe_eval(::NotTimeDependent, x::LazySum) = sum(_eval_at, x)
function safe_eval(::NotTimeDependent, x::LazySum, t::Number)
    throw(ArgumentError("attempting to evaluate time-independent LazySum at time"))
end

# For users
# using (t) should return NotTimeDependent LazySum
(x::LazySum)(t::Number) = safe_eval(x, t)
Base.sum(x::LazySum) = safe_eval(x) #so it works for untimedoperator

# we define the addition for LazySum and we do the rest with this
function Base.:+(SumOfOps1::LazySum, SumOfOps2::LazySum)
    return LazySum([SumOfOps1..., SumOfOps2...])
end

Base.:+(op1::LazySum, op2) = op1 + LazySum(op2)
Base.:+(op1, op2::LazySum) = LazySum(op1) + op2
Base.:+(op1::MultipliedOperator, op2::MultipliedOperator) = LazySum([op1, op2])

Base.repeat(x::LazySum, args...) = LazySum(repeat.(x, args...))
