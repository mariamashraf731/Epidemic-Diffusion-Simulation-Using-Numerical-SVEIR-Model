from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize

def impedance_function(omega):
    """
    Define the impedance function as a function of angular frequency (omega).
    """
    R = 225  # Resistance in Ohms
    C = 0.6e-6  # Capacitance in Farads
    L = 0.5  # Inductance in Henrys
    Z_target = 100  # Target impedance in Ohms
    
    term1 = 1 / R**2
    term2 = (omega * C - 1 / (omega * L))**2
    Z_inverse = np.sqrt(term1 + term2)
    return 1 / Z_inverse - Z_target

# Define the function f(x, y)
def f(x, y):
    return -8 * x + x**2 + 12 * y + 4 * y**2 - 2 * x * y

# Define the function to minimize
def f_min(xy):
    x, y = xy
    return -8 * x + x**2 + 12 * y + 4 * y**2 - 2 * x * y

# Define the function and its derivative
def func(y, z):
    A = (4 * DT * Co) / (Ro * (rc + tm)**2)
    B = 4 * DT / (V * rc**2)
    C = (2 * DT) / (rc * Ko)
    return (
        y**2 * np.log(y**2) - y**2 + 1
        - A
        + B * (y**2 - 1) * z
        + C * (y**2 - 1)
    )

def dfunc(y, z):
    B = 4 * DT / (V * rc**2)
    C = (2 * DT) / (rc * Ko)
    return (
        2 * y * np.log(y**2)
        + 2 * y
        - 2 * y
        + 2 * B * y * z
        + 2 * C * y
    )

# Define the function and its derivative
def func2(phi, eta):
    return (1 / phi) * (1 / np.tanh(3 * phi) - 1 / (3 * phi)) - eta

def dfunc2(phi):
    return (
        -1 / phi**2 * (1 / np.tanh(3 * phi) - 1 / (3 * phi))
        + 1 / phi * (-3 / np.sinh(3 * phi)**2 + 1 / (3 * phi**2))
    )

def system_func(x):
    """
    Define the system of equations and the Jacobian matrix.

    Parameters:
        x (numpy array): A vector [x, y].

    Returns:
        J (numpy array): The Jacobian matrix.
        f (numpy array): The function values.
    """
    # Define the system of equations
    f1 = x[0]**2 + x[1]**2 - 5  # x^2 = 5 - y^2
    f2 = x[0]**2 - x[1] - 1     # y + 1 = x^2
    f = np.array([f1, f2])

    # Define the Jacobian matrix
    J11 = 2 * x[0]  # df1/dx
    J12 = 2 * x[1]  # df1/dy
    J21 = 2 * x[0]  # df2/dx
    J22 = -1        # df2/dy
    J = np.array([[J11, J12], [J21, J22]])

    return J, f

def new_system_func(x):
    """
    Define the system of equations and the Jacobian matrix.

    Parameters:
        x (numpy array): A vector [x, y].

    Returns:
        J (numpy array): The Jacobian matrix.
        f (numpy array): The function values.
    """
    # Define the system of equations
    f1 = -x[0]**2 + x[0] + 0.5 - x[1]  # y = -x^2 + x + 0.5
    f2 = x[1] + 5 * x[0] * x[1] - x[0]**2  # y + 5xy = x^2
    f = np.array([f1, f2])

    # Define the Jacobian matrix
    J11 = -2 * x[0] + 1  # df1/dx
    J12 = -1  # df1/dy
    J21 = -2 * x[0] + 5 * x[1]  # df2/dx
    J22 = 1 + 5 * x[0]  # df2/dy
    J = np.array([[J11, J12], [J21, J22]])

    return J, f

def newtmult(func, x0, es=0.0001, maxit=50):
    """
    Newton-Raphson method for solving nonlinear systems of equations.

    Parameters:
        func (function): A function that returns the Jacobian matrix (J) and the function values (f).
        x0 (numpy array): Initial guess for the roots.
        es (float): Desired percent relative error (default = 0.0001%).
        maxit (int): Maximum allowable iterations (default = 50).

    Returns:
        x (numpy array): Vector of roots.
        f (numpy array): Vector of functions evaluated at the roots.
        ea (float): Approximate percent relative error.
        iter (int): Number of iterations performed.
    """
    iter = 0
    x = x0

    while True:
        # Evaluate the function and Jacobian at the current guess
        J, f = func(x)

        # Solve for the update vector dx
        dx = np.linalg.solve(J, -f)

        # Update the guess
        x = x + dx

        # Increment the iteration counter
        iter += 1

        # Calculate the approximate percent relative error
        ea = 100 * np.max(np.abs(dx / x))

        # Check for convergence or maximum iterations
        if iter >= maxit or ea <= es:
            break

    return x, f, ea, iter

if __name__ == "__main__":


    # ### Problem 6.19: Solving for Omega ###
    # # Initial guesses for omega
    # omega_guess1 = 1  # rad/s
    # omega_guess2 = 1000  # rad/s
    # # Solve for omega using fsolve
    # omega_solution_1 = fsolve(impedance_function, omega_guess1)[0]
    # omega_solution_2 = fsolve(impedance_function, omega_guess2)[0]

    # # Print the solutions
    # print("Solution for omega with guess 1:", omega_solution_1)
    # print("Solution for omega with guess 2:", omega_solution_2)


    # ### Problem 7.25: Graphical Solution ###
    # Create a meshgrid for x and y
    # x = np.linspace(-10, 10, 100)
    # y = np.linspace(-10, 10, 100)
    # X, Y = np.meshgrid(x, y)
    # Z = f(X, Y)

    # # Plot the 3D surface
    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    # ax.set_title('Surface Plot of f(x, y)')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('f(x, y)')
    # plt.show()




    # ### Problem 7.25: Numerical Solution ###
    # Create a meshgrid for x and y
    # Initial guess for (x, y)
    initial_guess = [0, 0]
    # Minimize the function
    result = minimize(f_min, initial_guess, method='Nelder-Mead')
    # Extract results
    x_min, y_min = result.x
    f_minimum = f_min(result.x)
    print("Minimum point (x, y):", (x_min, y_min))
    print("Minimum value of f(x, y):", f_minimum)
    
    ################ Problem 12.9 & 12.14 ##################
    # Initial guess
    x0 = np.array([1.5, 1.5])

    # Solve the system
    x, functions, ea, iter = newtmult(system_func, x0)

    # Display results
    print(f"Roots: [{x[0]:.4f}; {x[1]:.4f}]")
    print(f"Function values at roots: [{functions[0]:.4f}; {functions[1]:.4f}]")
    print(f"Approximate relative error: {ea:.4f}%")
    print(f"Iterations: {iter}")

    ################# Problem 12.8 #######################

    # Initial guess
    x0 = np.array([1.2, 1.2])

    # Solve the system
    x, functions, ea, iter = newtmult(new_system_func, x0)

    # Display results
    print(f"Roots: [{x[0]:.4f}; {x[1]:.4f}]")
    print(f"Function values at roots: [{functions[0]:.4f}; {functions[1]:.4f}]")
    print(f"Approximate relative error: {ea:.4f}%")
    print(f"Iterations: {iter}")

    ################## Problem 5.6 ###################
    # Given parameters
    DT = 8e-6  # cm^2/s
    V = 0.005  # cm/s
    rc = 0.0005  # cm
    tm = 5e-5  # cm
    Ko = 5.75e-5  # cm/s
    Co = 5  # µmole/cm^3
    Ro = 0.01  # µmole/(cm^3 s)

    # Initialize variables
    z_values = np.arange(0.001, 0.101, 0.01)  # cm
    rcrit_values = np.zeros_like(z_values)
    y0 = 1 # Initial guess for y

    # Newton-Raphson method
    for i, z in enumerate(z_values):
        y = y0
        for _ in range(100):  # Max iterations
            function = func(y, z)
            df = dfunc(y, z)
            y_new = y - function / df
            if abs(y_new - y) < 1e-6:  # Convergence criterion
                break
            y = y_new
        rcrit_values[i] = y * (rc + tm)

    # Plot the results
    plt.figure()
    plt.plot(z_values, rcrit_values, '-o')
    plt.xlabel('z (cm)')
    plt.ylabel('$r_{crit}$ (cm)')
    plt.title('Critical Distance $r_{crit}$ vs. z')
    plt.grid(True)
    plt.show()

    ################## Problem 5.7 ###################

    # Initialize variables
    eta_values = np.arange(0.30, 0.65, 0.05)  # Efficiency from 30% to 60%
    phi_values = np.zeros_like(eta_values)
    phi0 = 1.0  # Initial guess for phi

    # Newton-Raphson method
    for i, eta in enumerate(eta_values):
        phi = phi0
        for _ in range(100):  # Max iterations
            f_val = func2(phi, eta)
            df_val = dfunc2(phi)
            phi_new = phi - f_val / df_val
            if abs(phi_new - phi) < 1e-6:  # Convergence criterion
                break
            phi = phi_new
        phi_values[i] = phi

    # Plot the results
    plt.figure()
    plt.plot(eta_values, phi_values, '-o')
    plt.xlabel('Efficiency ($\eta$)')
    plt.ylabel('Thiele Modulus ($\phi$)')
    plt.title('Thiele Modulus vs. Efficiency')
    plt.grid(True)
    plt.show()