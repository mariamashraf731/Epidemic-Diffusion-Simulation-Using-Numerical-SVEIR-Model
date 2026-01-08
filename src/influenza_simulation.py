import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import root_scalar
from scipy.sparse import csr_matrix
from multiprocessing import Pool, cpu_count
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.special import digamma
from scipy.integrate import odeint, ode, solve_ivp
import time

def dss004(xl, xu, n, u):
    """
    Function dss004 computes the first derivative, ux, of a
    variable u over the spatial domain xl <= x <= xu from classical
    five-point, fourth-order finite difference approximations.

    Parameters:
    xl : float
        Lower boundary value of x (input)
    xu : float
        Upper boundary value of x (input)
    n : int
        Number of grid points in the x domain including the boundary points (input)
    u : numpy array
        One-dimensional array containing the values of u at the n grid points for which the derivative is to be computed (input)

    Returns:
    ux : numpy array
        One-dimensional array containing the numerical values of the derivatives of u at the n grid points (output)
    """

    # Compute the spatial increment
    dx = (xu - xl) / (n - 1)
    r4fdx = 1.0 / (12.0 * dx)
    nm2 = n - 2

    # Initialize the output array
    ux = np.zeros(n)

    # Equation (1) for the left end, point i = 1
    ux[0] = r4fdx * (-25.0 * u[0] + 48.0 * u[1] - 36.0 * u[2] + 16.0 * u[3] - 3.0 * u[4])

    # Equation (2) for the interior point, i = 2
    ux[1] = r4fdx * (-3.0 * u[0] - 10.0 * u[1] + 18.0 * u[2] - 6.0 * u[3] + 1.0 * u[4])

    # Equation (3) for the interior points, i = 3 to n-2
    for i in range(2, nm2):
        ux[i] = r4fdx * (1.0 * u[i-2] - 8.0 * u[i-1] + 0.0 * u[i] + 8.0 * u[i+1] - 1.0 * u[i+2])

    # Equation (4) for the interior point, i = n-1
    ux[n-2] = r4fdx * (-1.0 * u[n-5] + 6.0 * u[n-4] - 18.0 * u[n-3] + 10.0 * u[n-2] + 3.0 * u[n-1])

    # Equation (5) for the right end, point i = n
    ux[n-1] = r4fdx * (3.0 * u[n-5] - 16.0 * u[n-4] + 36.0 * u[n-3] - 48.0 * u[n-2] + 25.0 * u[n-1])

    return ux

def dss044(xl, xu, n, u, ux, nl, nu):
    """
    Function dss044 computes a fourth-order approximation of a
    second-order derivative, with or without the normal derivative
    at the boundary.

    Parameters:
    xl : float
        Left value of the spatial independent variable (input)
    xu : float
        Right value of the spatial independent variable (input)
    n : int
        Number of spatial grid points, including the end points (input)
    u : numpy array
        One-dimensional array of the dependent variable to be differentiated (input)
    ux : numpy array
        One-dimensional array of the first derivative of u (input)
    nl : int
        Integer index for the type of boundary condition at x = xl (input):
        1 - Dirichlet boundary condition at x = xl (ux(1) is not used)
        2 - Neumann boundary condition at x = xl (ux(1) is used)
    nu : int
        Integer index for the type of boundary condition at x = xu (input):
        1 - Dirichlet boundary condition at x = xu (ux(n) is not used)
        2 - Neumann boundary condition at x = xu (ux(n) is used)

    Returns:
    uxx : numpy array
        One-dimensional array of the second derivative of u (output)
    """

    # Compute the spatial increment
    dx = (xu - xl) / (n - 1)
    r12dxs = 1.0 / (12.0 * dx**2)

    # Initialize the output array
    uxx = np.zeros(n)

    # uxx at the left boundary
    if nl == 1:
        # Without ux (Dirichlet boundary condition)
        uxx[0] = r12dxs * (
            45.0 * u[0] - 154.0 * u[1] + 214.0 * u[2] - 156.0 * u[3] + 61.0 * u[4] - 10.0 * u[5]
        )
    elif nl == 2:
        # With ux (Neumann boundary condition)
        uxx[0] = r12dxs * (
            (-415.0 / 6.0) * u[0]
            + 96.0 * u[1]
            - 36.0 * u[2]
            + (32.0 / 3.0) * u[3]
            - (3.0 / 2.0) * u[4]
            - 50.0 * ux[0] * dx
        )

    # uxx at the right boundary
    if nu == 1:
        # Without ux (Dirichlet boundary condition)
        uxx[n-1] = r12dxs * (
            45.0 * u[n-1] - 154.0 * u[n-2] + 214.0 * u[n-3] - 156.0 * u[n-4] + 61.0 * u[n-5] - 10.0 * u[n-6]
        )
    elif nu == 2:
        # With ux (Neumann boundary condition)
        uxx[n-1] = r12dxs * (
            (-415.0 / 6.0) * u[n-1]
            + 96.0 * u[n-2]
            - 36.0 * u[n-3]
            + (32.0 / 3.0) * u[n-4]
            - (3.0 / 2.0) * u[n-5]
            + 50.0 * ux[n-1] * dx
        )

    # uxx at the interior grid points
    # i = 2
    uxx[1] = r12dxs * (
        10.0 * u[0] - 15.0 * u[1] - 4.0 * u[2] + 14.0 * u[3] - 6.0 * u[4] + 1.0 * u[5]
    )

    # i = n-1
    uxx[n-2] = r12dxs * (
        10.0 * u[n-1] - 15.0 * u[n-2] - 4.0 * u[n-3] + 14.0 * u[n-4] - 6.0 * u[n-5] + 1.0 * u[n-6]
    )

    # i = 3, 4,..., n-2
    for i in range(2, n - 2):
        uxx[i] = r12dxs * (
            -1.0 * u[i - 2] + 16.0 * u[i - 1] - 30.0 * u[i] + 16.0 * u[i + 1] - 1.0 * u[i + 2]
        )

    return uxx

def flu_1(t, u, parms):
    """
    Computes the time derivative vector of the S, V, E, I, R vectors.

    Args:
        t (float): Time variable (not used in computation but required for solver)
        u (numpy array): State vector containing S, V, E, I, R values
        parms (dict): Dictionary of parameters

    Returns:
        numpy array: Time derivative of the state vector
    """
    global ncall

    # Extract parameters from the input dictionary for readability
    beta, betae, betai, betav = parms["beta"], parms["betae"], parms["betai"], parms["betav"]
    alpha, phi, delta, theta, kappa = parms["alpha"], parms["phi"], parms["delta"], parms["theta"], parms["kappa"]
    sigma, gamma, r, d1, d2, d3, d4, d5 = parms["sigma"], parms["gamma"], parms["r"], parms["d1"], parms["d2"], parms["d3"], parms["d4"], parms["d5"]
    
    # Split the state vector into the five components: S, V, E, I, R
    nx = len(u) // 5  # Number of spatial points
    S, V, E, I, R = np.split(u, 5)  # Split the 1D state vector into five 1D arrays
    
    # Define spatial domain boundaries and Neumann boundary conditions
    xl, xu = -3, 3  # Domain boundaries in x
    nl, nu = 2, 2  # Neumann boundary condition specification
    
    # Compute the first spatial derivatives
    Sx, Vx, Ex, Ix, Rx = (
        dss004(xl, xu, nx, S),
        dss004(xl, xu, nx, V),
        dss004(xl, xu, nx, E),
        dss004(xl, xu, nx, I),
        dss004(xl, xu, nx, R)
    )
    
    # Apply Neumann boundary conditions: zero gradient at the boundaries
    Sx[0], Sx[nx-1] = 0, 0
    Vx[0], Vx[nx-1] = 0, 0
    Ex[0], Ex[nx-1] = 0, 0
    Ix[0], Ix[nx-1] = 0, 0
    Rx[0], Rx[nx-1] = 0, 0
    
    # Compute the second spatial derivatives (assume dss044 is a predefined function)
    Sxx, Vxx, Exx, Ixx, Rxx = (
        dss044(xl, xu, nx, S, Sx, nl, nu),
        dss044(xl, xu, nx, V, Vx, nl, nu),
        dss044(xl, xu, nx, E, Ex, nl, nu),
        dss044(xl, xu, nx, I, Ix, nl, nu),
        dss044(xl, xu, nx, R, Rx, nl, nu)
    )
    
    # Initialize arrays for the time derivatives of S, V, E, I, R
    St, Vt, Et, It, Rt = np.zeros(nx), np.zeros(nx), np.zeros(nx), np.zeros(nx), np.zeros(nx)
    
    # Compute the PDEs for each spatial point
    for i in range(nx):
        # Intermediate terms to simplify the equations
        ES, IS, EV, IV, IE, IR = (
            E[i] * S[i],  # E(x,t) * S(x,t)
            I[i] * S[i],  # I(x,t) * S(x,t)
            E[i] * V[i],  # E(x,t) * V(x,t)
            I[i] * V[i],  # I(x,t) * V(x,t)
            I[i] * E[i],  # I(x,t) * E(x,t)
            I[i] * R[i]   # I(x,t) * R(x,t)
        )
        
        # Equations for the time derivatives
        St[i] = -beta * betae * ES - beta * betai * IS + alpha * IS - phi * S[i] - r * S[i] + delta * R[i] + theta * V[i] + r + d1 * Sxx[i]
        Vt[i] = -beta * betae * betav * EV - beta * betai * betav * IV + alpha * IV - r * V[i] - theta * V[i] + phi * S[i] + d2 * Vxx[i]
        Et[i] = beta * betae * ES + beta * betai * IS + beta * betae * betav * EV + beta * betai * betav * IV + alpha * IE - (r + kappa + sigma) * E[i] + d3 * Exx[i]
        It[i] = sigma * E[i] - (r + alpha + gamma) * I[i] + alpha * (I[i] ** 2) + d4 * Ixx[i]
        Rt[i] = kappa * E[i] + gamma * I[i] - r * R[i] - delta * R[i] + alpha * IR + d5 * Rxx[i]
    
    # Concatenate the time derivative arrays into a single vector
    ut = np.concatenate([St, Vt, Et, It, Rt])
    
    # Increment the global counter for function calls
    ncall += 1
    
    return ut  # Return the time derivatives

# Load functions for analytical solutions and derivatives (assume they are defined)
# from flu_1 import flu_1
# from dss004 import dss004
# from dss044 import dss044

# Output format selection
ip = 1  # 1 for solutions vs x, 2 for solutions vs t at specific x

# Grid in x
nx = 61  # Number of spatial points
xl, xu = -3, 3  # Spatial domain boundaries
xg = np.linspace(xl, xu, nx)  # Create a grid of x values
#xg = np.linspace(xl, xu, (xu-xl)/(nx-1))  # Create a grid of x values
#xg = np.array([float(f"{x:.1f}") for x in xg])

# Grid in t
if ip == 1:
    nout, t0, tf = 11, 0, 60  # 11 time points for t = 0, 6, ..., 60
elif ip == 2:
    nout, t0, tf = 61, 0, 60  # 61 time points for smoother plots in t

tout = np.linspace(t0, tf, nout)  # Create a grid of time values
#tout = np.array([float(f"{t:.1f}") for t in tout])
# Parameters
beta, betae, betai, betav = 0.5140, 0.250, 1.0, 0.9  # Infection and transmission rates
sigma, gamma, delta = 1.0/2.0, 1.0/5.0, 1.0/365.0  # Rates for progression and recovery
mu, r, kappa, alpha = 5.50e-08, 1.140e-05, 1.857e-04, 9.30e-06  # Demographic and epidemiological parameters
theta, phi = 1.0/365.0, 1/20  # Immunization and waning rates
d1, d2, d3, d4, d5 = 0.05, 0.05, 0.025, 0.001, 0.0  # Diffusion coefficients

# Display selected parameters
print(f"\n\n betav = {betav:.3f} phi = {phi:.3f}\n")  # Display beta_v and phi values

# Initial conditions
u0 = np.zeros(5 * nx)  # Initialize the state vector
for ix in range(nx):
    u0[ix] = 0.86 * np.exp(-(xg[ix]/1.4)**2)  # Initial condition for S(x,t)
    u0[ix+nx] = 0.10 * np.exp(-(xg[ix]/1.4)**2)  # Initial condition for V(x,t)
    u0[ix+2*nx] = 0  # Initial condition for E(x,t)
    u0[ix+3*nx] = 0.04 * np.exp(-xg[ix]**2)  # Initial condition for I(x,t)
    u0[ix+4*nx] = 0  # Initial condition for R(x,t)

ncall = 0  # Counter for the number of calls to flu_1
rtol = 1e-6
atol = 1e-6

params = {
    "beta": beta, "betae": betae, "betai": betai, "betav": betav,
    "sigma": sigma, "gamma": gamma, "delta": delta, "mu": mu,
    "r": r, "kappa": kappa, "alpha": alpha, "theta": theta,
    "phi": phi, "d1": d1, "d2": d2, "d3": d3, "d4": d4, "d5": d5
}

#'''
# ODE integration 
out = odeint(flu_1, u0, tout, args=(params,), tfirst=True) 
#'''

'''
# Create DataFrame for output data
df_output = pd.DataFrame(out.T, columns=[str(int(t)) for t in tout])

# Add u0 as an extra column
df_output.insert(0, "SVEIR_Synth_Data", u0)  # Only first nx values (adjust if needed)

# Save to CSV
df_output.to_csv(f'SVEIR_data_ip{ip}.csv', index=False)
'''

'''
# Solve the ODE with parameters
out_2 = solve_ivp(flu_1, [t0, tf], u0, t_eval=tout, args=(params,),method='LSODA',atol=1e-8, rtol=1e-8,first_step=0.1, max_step=0.1)

# Extract solution
#t_values = out.t
out = out_2.y.T

# Parameters dictionary
params = {
    "beta": beta, "betae": betae, "betai": betai, "betav": betav,
    "sigma": sigma, "gamma": gamma, "delta": delta, "mu": mu,
    "r": r, "kappa": kappa, "alpha": alpha, "theta": theta,
    "phi": phi, "d1": d1, "d2": d2, "d3": d3, "d4": d4, "d5": d5
}

# Initialize the solver
solver = ode(flu_2)
solver.set_integrator("dopri5")  # Equivalent to RK45
solver.set_f_params(params)  # Pass parameters to the function
solver.set_initial_value(u0, t0)  # Set initial state and time

# Solve the ODE
solution = []
t_values = []
while solver.successful() and solver.t <= tf:
    solver.integrate(solver.t + (tf - t0) / (len(tout) - 1))
    t_values.append(solver.t)
    solution.append(solver.y)

# Convert to NumPy arrays
t_values = np.array(t_values)
solution = np.array(solution)
out = solution
'''
#'''
# Print dimensions of the output matrix
print(f"nrow(out) = {out.shape[0]}, ncol(out) = {out.shape[1]}")

# Arrays for plotting numerical solutions
S_xplot = np.zeros((nx, nout))  # Array for S(x,t)
V_xplot = np.zeros((nx, nout))  # Array for V(x,t)
E_xplot = np.zeros((nx, nout))  # Array for E(x,t)
I_xplot = np.zeros((nx, nout))  # Array for I(x,t)
R_xplot = np.zeros((nx, nout))  # Array for R(x,t)

# Extract data from the output matrix for plotting
for it in range(nout):
    for ix in range(nx):
        S_xplot[ix, it] = out[it, ix]  # Extract S(x,t)
        V_xplot[ix, it] = out[it, ix+nx]  # Extract V(x,t)
        E_xplot[ix, it] = out[it, ix+2*nx]  # Extract E(x,t)
        I_xplot[ix, it] = out[it, ix+3*nx]  # Extract I(x,t)
        R_xplot[ix, it] = out[it, ix+4*nx]  # Extract R(x,t)

# Display numerical solutions
if ip == 1:
    for it in range(nout):
        if it in [0, 10]:  # Only display results for t = 0 and t = 60
            print("\n\n t x S(x,t) V(x,t)")
            print("\n E(x,t) I(x,t) R(x,t)")
            for ix in range(nx):
                print(f"\n {tout[it]:6.1f} {xg[ix]:7.2f} {S_xplot[ix, it]:12.5f} {V_xplot[ix, it]:12.5f}")
                print(f" {E_xplot[ix, it]:14.5f} {I_xplot[ix, it]:12.5f} {R_xplot[ix, it]:12.5f}")

if ip == 2:
    for it in range(nout):
        if it in [0, 60]:
            print("\n\n t       x       S(x,t)      V(x,t)")
            print(" E(x,t)   I(x,t)      R(x,t)")
            for ix in range(nx):
                print(f"\n {tout[it]:6.1f}{xg[ix]:7.2f}{S_xplot[ix, it]:12.5f}{V_xplot[ix, it]:12.5f}")
                print(f"{E_xplot[ix, it]:14.5f}{I_xplot[ix, it]:12.5f}{R_xplot[ix, it]:12.5f}")

# Calls to ODE routine
print(f"\n\n ncall = {ncall:5d}\n\n")

#'''
# Plot S, V, E, I, R numerical solutions vs x with t as a parameter (t = 0,6,...,60)
if ip == 1:
    plt.figure(figsize=(6, 4))
    plt.plot(xg, S_xplot, linewidth=2)
    plt.xlabel("x")
    plt.ylabel("S(x,t), t=0,6,...,60")
    plt.title("S(x,t); t=0,6,...,60;")
    plt.xlim([xl, xu])
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(xg, V_xplot, linewidth=2)
    plt.xlabel("x")
    plt.ylabel("V(x,t), t=0,6,...,60")
    plt.title("V(x,t); t=0,6,...,60;")
    plt.xlim([xl, xu])
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(xg, E_xplot, linewidth=2)
    plt.xlabel("x")
    plt.ylabel("E(x,t), t=0,6,...,60")
    plt.title("E(x,t); t=0,6,...,60;")
    plt.xlim([xl, xu])
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(xg, I_xplot, linewidth=2)
    plt.xlabel("x")
    plt.ylabel("I(x,t), t=0,6,...,60")
    plt.title("I(x,t); t=0,6,...,60;")
    plt.xlim([xl, xu])
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(xg, R_xplot, linewidth=2)
    plt.xlabel("x")
    plt.ylabel("R(x,t), t=0,6,...,60")
    plt.title("R(x,t); t=0,6,...,60;")
    plt.xlim([xl, xu])
    plt.grid(True)
    plt.show()

# Plot S, V, E, I, R vs t at x = 0 (t = 0,1,...,60)
if ip == 2:
    S_tplot = np.zeros(nout)
    V_tplot = np.zeros(nout)
    E_tplot = np.zeros(nout)
    I_tplot = np.zeros(nout)
    R_tplot = np.zeros(nout)

    for it in range(nout):
        S_tplot[it] = S_xplot[30, it]  # Adjusted index (R is 1-based, Python is 0-based)
        V_tplot[it] = V_xplot[30, it]
        E_tplot[it] = E_xplot[30, it]
        I_tplot[it] = I_xplot[30, it]
        R_tplot[it] = R_xplot[30, it]

    plt.figure(figsize=(6, 4))
    plt.plot(tout, S_tplot, linewidth=2)
    plt.xlabel("t")
    plt.ylabel("S(x,t), x = 0")
    plt.title("S(x,t); x = 0")
    plt.xlim([t0, tf])
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(tout, V_tplot, linewidth=2)
    plt.xlabel("t")
    plt.ylabel("V(x,t), x = 0")
    plt.title("V(x,t); x = 0")
    plt.xlim([t0, tf])
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(tout, E_tplot, linewidth=2)
    plt.xlabel("t")
    plt.ylabel("E(x,t), x = 0")
    plt.title("E(x,t); x = 0")
    plt.xlim([t0, tf])
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(tout, I_tplot, linewidth=2)
    plt.xlabel("t")
    plt.ylabel("I(x,t), x = 0")
    plt.title("I(x,t); x = 0")
    plt.xlim([t0, tf])
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(tout, R_tplot, linewidth=2)
    plt.xlabel("t")
    plt.ylabel("R(x,t), x = 0")
    plt.title("R(x,t); x = 0")
    plt.xlim([t0, tf])
    plt.grid(True)
    plt.show()
#'''

############################################ FVM ##################################
def flu_1_fvm(t, u, parms):
    """
    Function flu_1_fvm computes the t derivative vector using FVM with Diffusion
    """
    global ncall

    # Extract parameters from the input dictionary for readability
    beta, betae, betai, betav = parms["beta"], parms["betae"], parms["betai"], parms["betav"]
    alpha, phi, delta, theta, kappa = parms["alpha"], parms["phi"], parms["delta"], parms["theta"], parms["kappa"]
    sigma, gamma, r, d1, d2, d3, d4, d5 = parms["sigma"], parms["gamma"], parms["r"], parms["d1"], parms["d2"], parms["d3"], parms["d4"], parms["d5"]
    
    # Split the state vector into the five components: S, V, E, I, R
    nx = len(u) // 5  # Number of spatial points
    S, V, E, I, R = np.split(u, 5)  # Split the 1D state vector into five 1D arrays
    
    # Define spatial domain boundaries and Neumann boundary conditions
    xl, xu = -3, 3  # Domain boundaries in x
    nl, nu = 2, 2  # Neumann boundary condition specification
   
    # Initialize the derivative vector
    dudt = np.zeros(5 * nx)

    # Calculate cell widths (assuming uniform grid)
    dx = (xu - xl) / (nx - 1)

    # --- Flux Calculations (Central Differencing for Diffusion Terms) ---

    # Function to approximate d^2var/dx^2 using FVM-like stencil
    def approximate_diffusion(var, d):
        diffusion_term = np.zeros(nx)  # Result vector
        for i in range(1, nx - 1):  # Loop from 1 to nx-2 (Python indexing)
            diffusion_term[i] = d * (var[i+1] - 2*var[i] + var[i-1]) / (dx**2)

        # Boundary conditions: zero flux dS/dx = 0
        diffusion_term[0] = d * (var[1] - var[0]) / (dx**2)  # Left boundary.
        diffusion_term[nx-1] = d * (var[nx-2] - var[nx-1]) / (dx**2)  # Right Boundary

        return diffusion_term

    # Calculate Diffusion terms for each variable
    diffusion_S = approximate_diffusion(S, d1)
    diffusion_V = approximate_diffusion(V, d2)
    diffusion_E = approximate_diffusion(E, d3)
    diffusion_I = approximate_diffusion(I, d4)
    diffusion_R = approximate_diffusion(R, d5)

    # --- Loop over cells and compute the change in each cell ---
    for i in range(nx):

        # Source/Sink terms (ODE part)
        ES = E[i] * S[i]
        IS = I[i] * S[i]
        EV = E[i] * V[i]
        IV = I[i] * V[i]
        IE = I[i] * E[i]
        IR = I[i] * R[i]

        S_source = -beta * betae * ES - beta * betai * IS + alpha * IS - phi * S[i] - r * S[i] + delta * R[i] + theta * V[i] + r
        V_source = -beta * betae * betav * EV - beta * betai * betav * IV + alpha * IV - r * V[i] - theta * V[i] + phi * S[i]
        E_source = beta * betae * ES + beta * betai * IS + beta * betae * betav * EV + beta * betai * betav * IV + alpha * IE - (r + kappa + sigma) * E[i]
        I_source = sigma * E[i] - (r + alpha + gamma) * I[i] + alpha * (I[i]**2)
        R_source = kappa * E[i] + gamma * I[i] - r * R[i] - delta * R[i] + alpha * IR


        # Update the derivatives - NOW INCLUDE DIFFUSION TERMS
        dudt[i] = S_source + diffusion_S[i]  # dS/dt
        dudt[i + nx] = V_source + diffusion_V[i]  # dV/dt
        dudt[i + 2 * nx] = E_source + diffusion_E[i]  # dE/dt
        dudt[i + 3 * nx] = I_source + diffusion_I[i]  # dI/dt
        dudt[i + 4 * nx] = R_source + diffusion_R[i]  # dR/dt

    return dudt  # Must return as a list
# Output format selection
ip = 1  # 1 for solutions vs x, 2 for solutions vs t at specific x

# Grid in x
nx = 61  # Number of spatial points
xl, xu = -3, 3  # Spatial domain boundaries
xg = np.linspace(xl, xu, nx,dtype=np.float64)  # Create a grid of x values
#xg = np.linspace(xl, xu, (xu-xl)/(nx-1))  # Create a grid of x values
#xg = np.array([float(f"{x:.1f}") for x in xg])

# Grid in t
if ip == 1:
    nout, t0, tf = 11, 0, 60  # 11 time points for t = 0, 6, ..., 60

elif ip == 2:
    nout, t0, tf = 61, 0, 60  # 61 time points for smoother plots in t

tout = np.linspace(t0, tf, nout, dtype=np.float64)  # Create a grid of time values
#tout = np.array([float(f"{t:.1f}") for t in tout])
# Parameters
beta, betae, betai, betav = 0.5140, 0.250, 1.0, 0.9  # Infection and transmission rates
sigma, gamma, delta = 1.0/2.0, 1.0/5.0, 1.0/365.0  # Rates for progression and recovery
mu, r, kappa, alpha = 5.50e-08, 1.140e-05, 1.857e-04, 9.30e-06  # Demographic and epidemiological parameters
theta, phi = 1.0/365.0, 1/20  # Immunization and waning rates
d1, d2, d3, d4, d5 = 0.05, 0.05, 0.025, 0.001, 0.0  # Diffusion coefficients

# Display selected parameters
print(f"\n\n betav = {betav:.3f} phi = {phi:.3f}\n")  # Display beta_v and phi values

# Initial conditions
u0 = np.zeros(5 * nx,dtype=np.float64)  # Initialize the state vector
for ix in range(nx):
    u0[ix] = 0.86 * np.exp(-(xg[ix]/1.4)**2)  # Initial condition for S(x,t)
    u0[ix+nx] = 0.10 * np.exp(-(xg[ix]/1.4)**2)  # Initial condition for V(x,t)
    u0[ix+2*nx] = 0  # Initial condition for E(x,t)
    u0[ix+3*nx] = 0.04 * np.exp(-xg[ix]**2)  # Initial condition for I(x,t)
    u0[ix+4*nx] = 0  # Initial condition for R(x,t)

'''
file_name = 'SVEIR_data_ip1_R.csv'
df = pd.read_csv(file_name)

# Step 3: Extract the second column (index 1, since Python uses 0-based indexing)
second_column = df.iloc[:, 1]

# Step 4: Convert to NumPy array
u0 = second_column.to_numpy()
'''
ncall = 0  # Counter for the number of calls to flu_1
rtol = 1e-6
atol = 1e-6

params = {
    "beta": beta, "betae": betae, "betai": betai, "betav": betav,
    "sigma": sigma, "gamma": gamma, "delta": delta, "mu": mu,
    "r": r, "kappa": kappa, "alpha": alpha, "theta": theta,
    "phi": phi, "d1": d1, "d2": d2, "d3": d3, "d4": d4, "d5": d5
}

'''

# Create DataFrame for output data
df_output = pd.DataFrame(out.T, columns=[str(int(t)) for t in tout])

# Add u0 as an extra column
df_output.insert(0, "SVEIR_Synth_Data", u0)  # Only first nx values (adjust if needed)

# Save to CSV
df_output.to_csv(f'SVEIR_data_ip{ip}.csv', index=False)
out = df.iloc[:, 2:].to_numpy().T
'''

'''
# Solve the ODE with parameters
out_2 = solve_ivp(flu_1, (t0, tf), u0, t_eval=tout, args=(params,),method='LSODA')

# Extract solution
#t_values = out.t
out = out_2.y.T
'''

# ODE integration 
start = time.time()
out = odeint(flu_1_fvm, u0, tout, args=(params,), tfirst=True, rtol=1e-8, atol=1e-8, h0=0.01, mxstep=5000) 
end = time.time()

print("FVM AVG Time = ", (end - start))

'''
# Parameters dictionary
params = {
    "beta": beta, "betae": betae, "betai": betai, "betav": betav,
    "sigma": sigma, "gamma": gamma, "delta": delta, "mu": mu,
    "r": r, "kappa": kappa, "alpha": alpha, "theta": theta,
    "phi": phi, "d1": d1, "d2": d2, "d3": d3, "d4": d4, "d5": d5
}

# Initialize the solver
solver = ode(flu_2)
solver.set_integrator("dopri5")  # Equivalent to RK45
solver.set_f_params(params)  # Pass parameters to the function
solver.set_initial_value(u0, t0)  # Set initial state and time

# Solve the ODE
solution = []
t_values = []
while solver.successful() and solver.t <= tf:
    solver.integrate(solver.t + (tf - t0) / (len(tout) - 1))
    t_values.append(solver.t)
    solution.append(solver.y)

# Convert to NumPy arrays
t_values = np.array(t_values)
solution = np.array(solution)
out = solution
'''

# Print dimensions of the output matrix
print(f"nrow(out) = {out.shape[0]}, ncol(out) = {out.shape[1]}")

# Arrays for plotting numerical solutions
V_xplot = np.zeros((nx, nout),dtype=np.float64)  # Array for V(x,t)
E_xplot = np.zeros((nx, nout),dtype=np.float64)  # Array for E(x,t)
S_xplot = np.zeros((nx, nout),dtype=np.float64)  # Array for S(x,t)
I_xplot = np.zeros((nx, nout),dtype=np.float64)  # Array for I(x,t)
R_xplot = np.zeros((nx, nout),dtype=np.float64)  # Array for R(x,t)

# Extract data from the output matrix for plotting
for it in range(nout):
    for ix in range(nx):
        S_xplot[ix, it] = out[it, ix]  # Extract S(x,t)
        V_xplot[ix, it] = out[it, ix+nx]  # Extract V(x,t)
        E_xplot[ix, it] = out[it, ix+2*nx]  # Extract E(x,t)
        I_xplot[ix, it] = out[it, ix+3*nx]  # Extract I(x,t)
        R_xplot[ix, it] = out[it, ix+4*nx]  # Extract R(x,t)

# Display numerical solutions
if ip == 1:
    for it in range(nout):
        if it in [0, 10]:  # Only display results for t = 0 and t = 60
            print("\n\n t x S(x,t) V(x,t)")
            print("\n E(x,t) I(x,t) R(x,t)")
            for ix in range(nx):
                print(f"\n {tout[it]:6.1f} {xg[ix]:7.2f} {S_xplot[ix, it]:12.5f} {V_xplot[ix, it]:12.5f}")
                print(f" {E_xplot[ix, it]:14.5f} {I_xplot[ix, it]:12.5f} {R_xplot[ix, it]:12.5f}")

if ip == 2:
    for it in range(nout):
        if it in [0, 60]:
            print("\n\n t       x       S(x,t)      V(x,t)")
            print(" E(x,t)   I(x,t)      R(x,t)")
            for ix in range(nx):
                print(f"\n {tout[it]:6.1f}{xg[ix]:7.2f}{S_xplot[ix, it]:12.5f}{V_xplot[ix, it]:12.5f}")
                print(f"{E_xplot[ix, it]:14.5f}{I_xplot[ix, it]:12.5f}{R_xplot[ix, it]:12.5f}")

# Calls to ODE routine
print(f"\n\n ncall = {ncall:5d}\n\n")

#'''
# Plot S, V, E, I, R numerical solutions vs x with t as a parameter (t = 0,6,...,60)
if ip == 1:
    plt.figure(figsize=(6, 4))
    plt.plot(xg, S_xplot, linewidth=2)
    plt.xlabel("x")
    plt.ylabel("S(x,t), t=0,6,...,60")
    plt.title("S(x,t); t=0,6,...,60;")
    plt.xlim([xl, xu])
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(xg, V_xplot, linewidth=2)
    plt.xlabel("x")
    plt.ylabel("V(x,t), t=0,6,...,60")
    plt.title("V(x,t); t=0,6,...,60;")
    plt.xlim([xl, xu])
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(xg, E_xplot, linewidth=2)
    plt.xlabel("x")
    plt.ylabel("E(x,t), t=0,6,...,60")
    plt.title("E(x,t); t=0,6,...,60;")
    plt.xlim([xl, xu])
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(xg, I_xplot, linewidth=2)
    plt.xlabel("x")
    plt.ylabel("I(x,t), t=0,6,...,60")
    plt.title("I(x,t); t=0,6,...,60;")
    plt.xlim([xl, xu])
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(xg, R_xplot, linewidth=2)
    plt.xlabel("x")
    plt.ylabel("R(x,t), t=0,6,...,60")
    plt.title("R(x,t); t=0,6,...,60;")
    plt.xlim([xl, xu])
    plt.grid(True)
    plt.show()

# Plot S, V, E, I, R vs t at x = 0 (t = 0,1,...,60)
if ip == 2:
    S_tplot = np.zeros(nout,dtype=np.float64)
    V_tplot = np.zeros(nout,dtype=np.float64)
    E_tplot = np.zeros(nout,dtype=np.float64)
    I_tplot = np.zeros(nout,dtype=np.float64)
    R_tplot = np.zeros(nout,dtype=np.float64)

    for it in range(nout):
        S_tplot[it] = S_xplot[30, it]  # Adjusted index (R is 1-based, Python is 0-based)
        V_tplot[it] = V_xplot[30, it]
        E_tplot[it] = E_xplot[30, it]
        I_tplot[it] = I_xplot[30, it]
        R_tplot[it] = R_xplot[30, it]

    plt.figure(figsize=(6, 4))
    plt.plot(tout, S_tplot, linewidth=2)
    plt.xlabel("t")
    plt.ylabel("S(x,t), x = 0")
    plt.title("S(x,t); x = 0")
    plt.xlim([t0, tf])
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(tout, V_tplot, linewidth=2)
    plt.xlabel("t")
    plt.ylabel("V(x,t), x = 0")
    plt.title("V(x,t); x = 0")
    plt.xlim([t0, tf])
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(tout, E_tplot, linewidth=2)
    plt.xlabel("t")
    plt.ylabel("E(x,t), x = 0")
    plt.title("E(x,t); x = 0")
    plt.xlim([t0, tf])
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(tout, I_tplot, linewidth=2)
    plt.xlabel("t")
    plt.ylabel("I(x,t), x = 0")
    plt.title("I(x,t); x = 0")
    plt.xlim([t0, tf])
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(tout, R_tplot, linewidth=2)
    plt.xlabel("t")
    plt.ylabel("R(x,t), x = 0")
    plt.title("R(x,t); x = 0")
    plt.xlim([t0, tf])
    plt.grid(True)
    plt.show()

###################################### Euler ######################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import root_scalar
from scipy.sparse import csr_matrix
from multiprocessing import Pool, cpu_count
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.special import digamma
from scipy.integrate import odeint, ode, solve_ivp

def euler_first_derivative(xl, xu, n, u):
    """
    Compute the first derivative using Euler's method.
    """
    dx = (xu - xl) / (n - 1)
    ux = np.zeros(n)

    # Forward difference for interior points
    for i in range(n - 1):
        ux[i] = (u[i + 1] - u[i]) / dx

    # Boundary condition for last point (backward difference)
    ux[n - 1] = (u[n - 1] - u[n - 2]) / dx

    return ux

def euler_second_derivative(xl, xu, n, u):
    """
    Compute the second derivative using a central difference approximation.
    """
    dx = (xu - xl) / (n - 1)
    uxx = np.zeros(n)

    # Central difference for interior points
    for i in range(1, n - 1):
        uxx[i] = (u[i + 1] - 2 * u[i] + u[i - 1]) / dx**2

    # Boundary conditions (set second derivative to zero at boundaries)
    uxx[0] = 0
    uxx[n - 1] = 0

    return uxx

def flu_1_Euler(t, u, parms):
    """
    Computes the time derivative vector of the S, V, E, I, R vectors.

    Args:
        t (float): Time variable (not used in computation but required for solver)
        u (numpy array): State vector containing S, V, E, I, R values
        parms (dict): Dictionary of parameters

    Returns:
        numpy array: Time derivative of the state vector
    """
    global ncall

    # Extract parameters from the input dictionary for readability
    beta, betae, betai, betav = parms["beta"], parms["betae"], parms["betai"], parms["betav"]
    alpha, phi, delta, theta, kappa = parms["alpha"], parms["phi"], parms["delta"], parms["theta"], parms["kappa"]
    sigma, gamma, r, d1, d2, d3, d4, d5 = parms["sigma"], parms["gamma"], parms["r"], parms["d1"], parms["d2"], parms["d3"], parms["d4"], parms["d5"]

    # Split the state vector into the five components: S, V, E, I, R
    nx = len(u) // 5  # Number of spatial points
    S, V, E, I, R = np.split(u, 5)  # Split the 1D state vector into five 1D arrays

    # Define spatial domain boundaries and Neumann boundary conditions
    xl, xu = -3, 3  # Domain boundaries in x
    nl, nu = 2, 2  # Neumann boundary condition specification

    # Compute the first spatial derivatives
    Sx, Vx, Ex, Ix, Rx = (
        euler_first_derivative(xl, xu, nx, S),
        euler_first_derivative(xl, xu, nx, V),
        euler_first_derivative(xl, xu, nx, E),
        euler_first_derivative(xl, xu, nx, I),
        euler_first_derivative(xl, xu, nx, R)
    )

    # Apply Neumann boundary conditions: zero gradient at the boundaries
    Sx[0], Sx[nx-1] = 0, 0
    Vx[0], Vx[nx-1] = 0, 0
    Ex[0], Ex[nx-1] = 0, 0
    Ix[0], Ix[nx-1] = 0, 0
    Rx[0], Rx[nx-1] = 0, 0

    # Compute the second spatial derivatives (assume dss044 is a predefined function)
    Sxx, Vxx, Exx, Ixx, Rxx = (
        euler_second_derivative(xl, xu, nx, S),
        euler_second_derivative(xl, xu, nx, V),
        euler_second_derivative(xl, xu, nx, E),
        euler_second_derivative(xl, xu, nx, I),
        euler_second_derivative(xl, xu, nx, R)
    )

    # Initialize arrays for the time derivatives of S, V, E, I, R
    St, Vt, Et, It, Rt = np.zeros(nx), np.zeros(nx), np.zeros(nx), np.zeros(nx), np.zeros(nx)

    # Compute the PDEs for each spatial point
    for i in range(nx):
        # Intermediate terms to simplify the equations
        ES, IS, EV, IV, IE, IR = (
            E[i] * S[i],  # E(x,t) * S(x,t)
            I[i] * S[i],  # I(x,t) * S(x,t)
            E[i] * V[i],  # E(x,t) * V(x,t)
            I[i] * V[i],  # I(x,t) * V(x,t)
            I[i] * E[i],  # I(x,t) * E(x,t)
            I[i] * R[i]   # I(x,t) * R(x,t)
        )

        # Equations for the time derivatives
        St[i] = -beta * betae * ES - beta * betai * IS + alpha * IS - phi * S[i] - r * S[i] + delta * R[i] + theta * V[i] + r + d1 * Sxx[i]
        Vt[i] = -beta * betae * betav * EV - beta * betai * betav * IV + alpha * IV - r * V[i] - theta * V[i] + phi * S[i] + d2 * Vxx[i]
        Et[i] = beta * betae * ES + beta * betai * IS + beta * betae * betav * EV + beta * betai * betav * IV + alpha * IE - (r + kappa + sigma) * E[i] + d3 * Exx[i]
        It[i] = sigma * E[i] - (r + alpha + gamma) * I[i] + alpha * (I[i] ** 2) + d4 * Ixx[i]
        Rt[i] = kappa * E[i] + gamma * I[i] - r * R[i] - delta * R[i] + alpha * IR + d5 * Rxx[i]

    # Concatenate the time derivative arrays into a single vector
    ut = np.concatenate([St, Vt, Et, It, Rt])

    # Increment the global counter for function calls
    ncall += 1

    return ut  # Return the time derivatives

# Output format selection
ip = 1  # 1 for solutions vs x, 2 for solutions vs t at specific x

# Grid in x
nx = 61  # Number of spatial points
xl, xu = -3, 3  # Spatial domain boundaries
xg = np.linspace(xl, xu, nx)  # Create a grid of x values
#xg = np.linspace(xl, xu, (xu-xl)/(nx-1))  # Create a grid of x values
#xg = np.array([float(f"{x:.1f}") for x in xg])

# Grid in t
if ip == 1:
    nout, t0, tf = 11, 0, 60  # 11 time points for t = 0, 6, ..., 60
elif ip == 2:
    nout, t0, tf = 61, 0, 60  # 61 time points for smoother plots in t

tout = np.linspace(t0, tf, nout)  # Create a grid of time values
#tout = np.array([float(f"{t:.1f}") for t in tout])
# Parameters
beta, betae, betai, betav = 0.5140, 0.250, 1.0, 0.9  # Infection and transmission rates
sigma, gamma, delta = 1.0/2.0, 1.0/5.0, 1.0/365.0  # Rates for progression and recovery
mu, r, kappa, alpha = 5.50e-08, 1.140e-05, 1.857e-04, 9.30e-06  # Demographic and epidemiological parameters
theta, phi = 1.0/365.0, 1/20  # Immunization and waning rates
d1, d2, d3, d4, d5 = 0.05, 0.05, 0.025, 0.001, 0.0  # Diffusion coefficients

# Display selected parameters
print(f"\n\n betav = {betav:.3f} phi = {phi:.3f}\n")  # Display beta_v and phi values

# Initial conditions
u0 = np.zeros(5 * nx)  # Initialize the state vector
for ix in range(nx):
    u0[ix] = 0.86 * np.exp(-(xg[ix]/1.4)**2)  # Initial condition for S(x,t)
    u0[ix+nx] = 0.10 * np.exp(-(xg[ix]/1.4)**2)  # Initial condition for V(x,t)
    u0[ix+2*nx] = 0  # Initial condition for E(x,t)
    u0[ix+3*nx] = 0.04 * np.exp(-xg[ix]**2)  # Initial condition for I(x,t)
    u0[ix+4*nx] = 0  # Initial condition for R(x,t)

ncall = 0  # Counter for the number of calls to flu_1
rtol = 1e-6
atol = 1e-6

params = {
    "beta": beta, "betae": betae, "betai": betai, "betav": betav,
    "sigma": sigma, "gamma": gamma, "delta": delta, "mu": mu,
    "r": r, "kappa": kappa, "alpha": alpha, "theta": theta,
    "phi": phi, "d1": d1, "d2": d2, "d3": d3, "d4": d4, "d5": d5
}

#'''
start = time.time()
out = odeint(flu_1_Euler, u0, tout, args=(params,), tfirst=True, rtol=1e-8, atol=1e-8, h0=0.01, mxstep=5000) 
end = time.time()

print("Euler AVG Time = ", (end - start))
#'''

'''
# Create DataFrame for output data
df_output = pd.DataFrame(out.T, columns=[str(int(t)) for t in tout])

# Add u0 as an extra column
df_output.insert(0, "SVEIR_Synth_Data", u0)  # Only first nx values (adjust if needed)

# Save to CSV
df_output.to_csv(f'SVEIR_data_ip{ip}.csv', index=False)
'''

'''
# Solve the ODE with parameters
out_2 = solve_ivp(flu_1, [t0, tf], u0, t_eval=tout, args=(params,),method='LSODA',atol=1e-8, rtol=1e-8,first_step=0.1, max_step=0.1)

# Extract solution
#t_values = out.t
out = out_2.y.T

# Parameters dictionary
params = {
    "beta": beta, "betae": betae, "betai": betai, "betav": betav,
    "sigma": sigma, "gamma": gamma, "delta": delta, "mu": mu,
    "r": r, "kappa": kappa, "alpha": alpha, "theta": theta,
    "phi": phi, "d1": d1, "d2": d2, "d3": d3, "d4": d4, "d5": d5
}

# Initialize the solver
solver = ode(flu_2)
solver.set_integrator("dopri5")  # Equivalent to RK45
solver.set_f_params(params)  # Pass parameters to the function
solver.set_initial_value(u0, t0)  # Set initial state and time

# Solve the ODE
solution = []
t_values = []
while solver.successful() and solver.t <= tf:
    solver.integrate(solver.t + (tf - t0) / (len(tout) - 1))
    t_values.append(solver.t)
    solution.append(solver.y)

# Convert to NumPy arrays
t_values = np.array(t_values)
solution = np.array(solution)
out = solution
'''
#'''
# Print dimensions of the output matrix
print(f"nrow(out) = {out.shape[0]}, ncol(out) = {out.shape[1]}")

# Arrays for plotting numerical solutions
S_xplot = np.zeros((nx, nout))  # Array for S(x,t)
V_xplot = np.zeros((nx, nout))  # Array for V(x,t)
E_xplot = np.zeros((nx, nout))  # Array for E(x,t)
I_xplot = np.zeros((nx, nout))  # Array for I(x,t)
R_xplot = np.zeros((nx, nout))  # Array for R(x,t)

# Extract data from the output matrix for plotting
for it in range(nout):
    for ix in range(nx):
        S_xplot[ix, it] = out[it, ix]  # Extract S(x,t)
        V_xplot[ix, it] = out[it, ix+nx]  # Extract V(x,t)
        E_xplot[ix, it] = out[it, ix+2*nx]  # Extract E(x,t)
        I_xplot[ix, it] = out[it, ix+3*nx]  # Extract I(x,t)
        R_xplot[ix, it] = out[it, ix+4*nx]  # Extract R(x,t)

# Display numerical solutions
if ip == 1:
    for it in range(nout):
        if it in [0, 10]:  # Only display results for t = 0 and t = 60
            print("\n\n t x S(x,t) V(x,t)")
            print("\n E(x,t) I(x,t) R(x,t)")
            for ix in range(nx):
                print(f"\n {tout[it]:6.1f} {xg[ix]:7.2f} {S_xplot[ix, it]:12.5f} {V_xplot[ix, it]:12.5f}")
                print(f" {E_xplot[ix, it]:14.5f} {I_xplot[ix, it]:12.5f} {R_xplot[ix, it]:12.5f}")

if ip == 2:
    for it in range(nout):
        if it in [0, 60]:
            print("\n\n t       x       S(x,t)      V(x,t)")
            print(" E(x,t)   I(x,t)      R(x,t)")
            for ix in range(nx):
                print(f"\n {tout[it]:6.1f}{xg[ix]:7.2f}{S_xplot[ix, it]:12.5f}{V_xplot[ix, it]:12.5f}")
                print(f"{E_xplot[ix, it]:14.5f}{I_xplot[ix, it]:12.5f}{R_xplot[ix, it]:12.5f}")

# Calls to ODE routine
print(f"\n\n ncall = {ncall:5d}\n\n")

#'''
# Plot S, V, E, I, R numerical solutions vs x with t as a parameter (t = 0,6,...,60)
if ip == 1:
    plt.figure(figsize=(6, 4))
    plt.plot(xg, S_xplot, linewidth=2)
    plt.xlabel("x")
    plt.ylabel("S(x,t), t=0,6,...,60")
    plt.title("S(x,t); t=0,6,...,60;")
    plt.xlim([xl, xu])
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(xg, V_xplot, linewidth=2)
    plt.xlabel("x")
    plt.ylabel("V(x,t), t=0,6,...,60")
    plt.title("V(x,t); t=0,6,...,60;")
    plt.xlim([xl, xu])
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(xg, E_xplot, linewidth=2)
    plt.xlabel("x")
    plt.ylabel("E(x,t), t=0,6,...,60")
    plt.title("E(x,t); t=0,6,...,60;")
    plt.xlim([xl, xu])
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(xg, I_xplot, linewidth=2)
    plt.xlabel("x")
    plt.ylabel("I(x,t), t=0,6,...,60")
    plt.title("I(x,t); t=0,6,...,60;")
    plt.xlim([xl, xu])
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(xg, R_xplot, linewidth=2)
    plt.xlabel("x")
    plt.ylabel("R(x,t), t=0,6,...,60")
    plt.title("R(x,t); t=0,6,...,60;")
    plt.xlim([xl, xu])
    plt.grid(True)
    plt.show()

# Plot S, V, E, I, R vs t at x = 0 (t = 0,1,...,60)
if ip == 2:
    S_tplot = np.zeros(nout)
    V_tplot = np.zeros(nout)
    E_tplot = np.zeros(nout)
    I_tplot = np.zeros(nout)
    R_tplot = np.zeros(nout)

    for it in range(nout):
        S_tplot[it] = S_xplot[30, it]  # Adjusted index (R is 1-based, Python is 0-based)
        V_tplot[it] = V_xplot[30, it]
        E_tplot[it] = E_xplot[30, it]
        I_tplot[it] = I_xplot[30, it]
        R_tplot[it] = R_xplot[30, it]

    plt.figure(figsize=(6, 4))
    plt.plot(tout, S_tplot, linewidth=2)
    plt.xlabel("t")
    plt.ylabel("S(x,t), x = 0")
    plt.title("S(x,t); x = 0")
    plt.xlim([t0, tf])
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(tout, V_tplot, linewidth=2)
    plt.xlabel("t")
    plt.ylabel("V(x,t), x = 0")
    plt.title("V(x,t); x = 0")
    plt.xlim([t0, tf])
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(tout, E_tplot, linewidth=2)
    plt.xlabel("t")
    plt.ylabel("E(x,t), x = 0")
    plt.title("E(x,t); x = 0")
    plt.xlim([t0, tf])
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(tout, I_tplot, linewidth=2)
    plt.xlabel("t")
    plt.ylabel("I(x,t), x = 0")
    plt.title("I(x,t); x = 0")
    plt.xlim([t0, tf])
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(tout, R_tplot, linewidth=2)
    plt.xlabel("t")
    plt.ylabel("R(x,t), x = 0")
    plt.title("R(x,t); x = 0")
    plt.xlim([t0, tf])
    plt.grid(True)
    plt.show()