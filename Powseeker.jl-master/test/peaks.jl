using Powseeker
using ForwardDiff
using StaticArrays

n = 1000
rng = MersenneTwister(1)
r = [rand(rng, SVector{2,Float64}) for i in 1:n]

for i in 1:n
    if norm(grad_peaks(r[i]) - ForwardDiff.gradient(peaks, r[i])) > 0.0000001
        println(grad_peaks(r[i]))
        println(ForwardDiff.gradient(peaks, r[i]))
    end
end

@time for i in 1:n
    grad_peaks(r[i])
end

@time for i in 1:n
    ForwardDiff.gradient(peaks, r[i])
end

for i in 1:n
    if norm(grad_scaled_peaks(r[i]) - ForwardDiff.gradient(scaled_peaks, r[i])) > 0.0000001
        println(grad_scaled_peaks(r[i]))
        println(ForwardDiff.gradient(scaled_peaks, r[i]))
    end
end


for i in 1:n
    if norm(grad_peaks2(r[i]) - ForwardDiff.gradient(peaks2, r[i])) > 0.0000001
        @show grad_peaks2(r[i])
        @show ForwardDiff.gradient(peaks2, r[i])
    end
end

@time for i in 1:n
    grad_peaks2(r[i])
end

@time for i in 1:n
    ForwardDiff.gradient(peaks2, r[i])
end

for i in 1:n
    if norm(grad_scaled_peaks2(r[i]) - ForwardDiff.gradient(scaled_peaks2, r[i])) > 0.0000001
        @show grad_scaled_peaks2(r[i])
        @show ForwardDiff.gradient(scaled_peaks2, r[i])
    end
end

