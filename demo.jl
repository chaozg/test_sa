using DifferentialEquations
using Zygote, SciMLSensitivity
using ForwardDiff
using Plots
using NPZ
using LinearAlgebra

S = npzread("S.npy")
M = npzread("M.npy")
no_of_elements = 128
ode_reltol = 1e-8
ode_abstol = 1e-8

###### 2. ODE problem with mass matrix
function dudt!(du, u, p, t)
    du .= S*u
    du[1] = u[1]
    nothing
end

u0 = ones(no_of_elements+1)

f = ODEFunction(dudt!, mass_matrix = M)
tspan =(0.0, 1.0)

# 1. forward simulation and plot solution
prob = ODEProblem(f, u0, tspan)
sol = solve(prob, FBDF(), reltol = ode_reltol, abstol = ode_abstol, saveat = 0.1) #Rodas5 #FBDF

for i in 1:length(sol.t)
    sleep(0.2)
    display(plot(sol.u[i], label="t=$(round(sol.t[i], digits=3))", ylims=(0,1.1), framestyle = :box))
end

# 2. sensitivity
prob = ODEProblem(f, u0, tspan)
function fun_of_interest(u)
    _prob = remake(prob, u0 = u)
    sol = solve(_prob, FBDF(autodiff=false), reltol = ode_reltol, abstol = ode_abstol) #Rodas5 #FBDF
    return norm(sol.u[end])
end

# reverse mode
rs_jac = Zygote.gradient(fun_of_interest, u0)[1]
println(rs_jac)

# forward mode
fs_jac = ForwardDiff.gradient(fun_of_interest, u0)
println(fs_jac)

# plot sensitivity
plot(rs_jac, label="reverse", framestyle = :box)
plot!(fs_jac, label="forward", framestyle = :box)
