using DifferentialEquations
using Zygote, SciMLSensitivity
using ForwardDiff
using Plots
using NPZ
using LinearAlgebra

# 1. Parameters
S = npzread("S.npy")
M = npzread("M.npy")
no_of_elements = 128
ode_reltol = 1e-8
ode_abstol = 1e-8

# 2. ODE problem with mass matrix
function dudt!(du, u, p, t)
    du .= S*u
    du[1] = u[1]
    nothing
end

u0 = ones(no_of_elements+1)
f = ODEFunction(dudt!, mass_matrix = M)
tspan =(0.0, 1.0)
prob = ODEProblem(f, u0, tspan)

# 3. Forward simulation and plot solution
sol = solve(prob, FBDF(), reltol = ode_reltol, abstol = ode_abstol, saveat = 0.1) #Rodas5 #FBDF

for i in 1:length(sol.t)
    sleep(0.2)
    display(plot(sol.u[i], label="t=$(round(sol.t[i], digits=3))", ylims=(0,1.1), framestyle = :box))
end

# 4. Sensitivity of scaler function of interest
function scaler_fun_of_interest(u)
    _prob = remake(prob, u0 = u)
    sol = solve(_prob, FBDF(autodiff=false), reltol = ode_reltol, abstol = ode_abstol) #Rodas5 #FBDF
    return norm(sol.u[end])
end

# reverse mode
rs_grad = Zygote.gradient(scaler_fun_of_interest, u0)[1]

# forward mode
fs_grad = ForwardDiff.gradient(scaler_fun_of_interest, u0)

# plot gradient
scalar_plot = plot(rs_grad, label="reverse", framestyle = :box)
plot!(fs_grad, label="forward", framestyle = :box)
title!("Gradient of norm of solution")
display(scalar_plot)

# 5. Sensitivity of vector function of interest
function vector_fun_of_interest(u)
    _prob = remake(prob, u0 = u)
    sol = solve(_prob, FBDF(autodiff=false), reltol = ode_reltol, abstol = ode_abstol) #Rodas5 #FBDF
    return sol.u[end][1:2:end]
end

# reverse mode
rs_jac = Zygote.jacobian(vector_fun_of_interest, u0)[1]

# forward mode
fs_jac = ForwardDiff.jacobian(vector_fun_of_interest, u0)

# plot jacobian
plots_layout = @layout [a b; c d]
p1 = heatmap(rs_jac, label="reverse", framestyle = :box)
title!("reverse")
p2 = heatmap(fs_jac, label="forward", framestyle = :box)
title!("forward")
p3 = heatmap(rs_jac[:,2:end], label="reverse", framestyle = :box)
title!("reverse (without 1st col)")
p4 = heatmap(fs_jac[:,2:end], label="forward", framestyle = :box)
title!("forward (without 1st col)")
vector_plot = plot(p1, p2, p3, p4, layout = plots_layout)
display(vector_plot)
