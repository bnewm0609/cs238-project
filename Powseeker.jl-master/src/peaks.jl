

peaks(x::AbstractArray,y::AbstractArray) = 3*(1-x).^2.*exp(-(x.^2) - (y+1).^2) - 10*(x/5 - x.^3 - y.^5).*exp(-x.^2-y.^2) - 1/3*exp(-(x+1).^2 - y.^2) 
peaks(x::Number,y::Number) = 3*(1-x)^2*exp(-(x^2) - (y+1)^2) - 10*(x/5 - x^3 - y^5)*exp(-x^2-y^2) - 1/3*exp(-(x+1)^2 - y^2) 
peaks(v::AbstractVector) = peaks(v[1],v[2])

Power(a,b) = a^b
E = e
grad_peaks(x::Number, y::Number) = SVector((-2*Power(E,-2*x - Power(x,2) - Power(1 + y,2))* (-(Power(E,2*y)*(1 + x)) + 9*Power(E,2*x)*(1 - 2*Power(x,2) + Power(x,3)) + Power(E,1 + 2*x + 2*y)*(3 - 51*Power(x,2) + 30*Power(x,4) + 30*x*Power(y,5))))/3., (-2*Power(E,-2*x - Power(x,2) - Power(1 + y,2))* (-(Power(E,2*y)*y) + 9*Power(E,2*x)*Power(-1 + x,2)*(1 + y) + 3*Power(E,1 + 2*x + 2*y)*y*(-2*x + 10*Power(x,3) + 5*Power(y,3)*(-5 + 2*Power(y,2)))))/3.)
grad_peaks(v::AbstractVector) = grad_peaks(v[1], v[2])

grad_scaled_peaks(x, y) = 0.1*grad_peaks(x/1000, y/1000)
grad_scaled_peaks(v::AbstractVector) = 0.1*grad_peaks(v/1000)
scaled_peaks(x,y) = 100*peaks(x/1000, y/1000)
scaled_peaks(v::AbstractVector) = 100*peaks(v/1000)

function plot_peaks(n=100)
    x = linspace(-4, 4, n)
    y = linspace(-4, 4, n)
    surface(x,y,peaks)
end

function plot_scaled_peaks(n=100)
    x = linspace(-4000, 4000, n)
    y = linspace(-4000, 4000, n)
    surface(x,y,scaled_peaks, ratio=:equal)
end


peaks2(x,y) = 3*(1-x)^2*exp(-(x^2) - (y+1)^2) - 7*(x/5 - x^2 - y^4)*exp(-x^2/2-(y)^2) - 1/3*exp(-(x+1)^2 - y^2) 
peaks2(v) = peaks2(v[1], v[2])
scaled_peaks2(x,y) = 70*peaks2(x/1000, y/1000)
scaled_peaks2(v) = scaled_peaks2(v[1], v[2])

grad_peaks2(x,y) = SVector((exp(-2*x - Power(x,2) - Power(1 + y,2))*
      (10*exp(2*y)*(1 + x) - 90*exp(2*x)*(1 - 2*Power(x,2) + Power(x,3)) - 
        21*exp(1 + 2*x + Power(x,2)/2. + 2*y)*
         (1 - Power(x,2) + 5*Power(x,3) + 5*x*(-2 + Power(y,4)))))/15.,
   (2*exp(-Power(1 + x,2) - Power(y,2))*y)/3. + 
    28*exp(-Power(x,2)/2. - Power(y,2))*Power(y,3) - 
    6*exp(-Power(x,2) - Power(1 + y,2))*Power(-1 + x,2)*(1 + y) - 
    (14*exp(-Power(x,2)/2. - Power(y,2))*y*(-x + 5*Power(x,2) + 5*Power(y,4)))/5.)
grad_peaks2(v) = grad_peaks2(v[1], v[2])

grad_scaled_peaks2(v) = 0.07*grad_peaks2(v[1]/1000, v[2]/1000)

#=
grad_peaks2(x,y) = SVector((Power(E,-2*x - Power(x,2) - Power(1 + y,2))*
      (10*Power(E,2*y)*(1 + x) - 90*Power(E,2*x)*(1 - 2*Power(x,2) + Power(x,3)) - 
        21*Power(E,1 + 2*x + Power(x,2)/2. + 2*y)*
         (1 - Power(x,2) + 5*Power(x,3) + 5*x*(-2 + Power(y,4)))))/15.,
   (2*Power(E,-Power(1 + x,2) - Power(y,2))*y)/3. + 
    28*Power(E,-Power(x,2)/2. - Power(y,2))*Power(y,3) - 
    6*Power(E,-Power(x,2) - Power(1 + y,2))*Power(-1 + x,2)*(1 + y) - 
    (14*Power(E,-Power(x,2)/2. - Power(y,2))*y*(-x + 5*Power(x,2) + 5*Power(y,4)))/5.)
=#
