import numpy as np
from scipy.linalg import hilbert
from numpy.linalg import cond
from scipy.linalg import eig
from numpy.linalg import cond
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt
from scipy.linalg import lu  # LU decomposition from scipy

def GaussSeidelR(A, b, lambda_relax=1.0, es=1e-4, maxit=50):
    """
    Gauss-Seidel method with relaxation for solving Ax = b.

    Parameters:
        A (ndarray): Coefficient matrix (n x n).
        b (ndarray): Right-hand side vector (n x 1).
        lambda_relax (float): Relaxation factor (default is 1.0).
        es (float): Stopping criterion for approximate error (default is 0.0001).
        maxit (int): Maximum number of iterations (default is 50).

    Returns:
        x (ndarray): Solution vector.
        ea (ndarray): Approximate error at each iteration.
        iter_count (int): Number of iterations performed.
    """
    n = len(b)
    x = np.zeros(n)  # Initialize solution vector
    ea = np.full(n, np.inf)  # Initialize approximate errors
    iter_count = 0  # Initialize iteration counter

    while True:
        x_old = x.copy()  # Store the previous iteration values
        for i in range(n):
            # Compute the summation term
            sum1 = np.dot(A[i, :i], x[:i])  # Terms before x_i
            sum2 = np.dot(A[i, i+1:], x_old[i+1:])  # Terms after x_i
            x_new = (b[i] - sum1 - sum2) / A[i, i]  # Update x_i without relaxation
            # Apply relaxation
            x[i] = lambda_relax * x_new + (1 - lambda_relax) * x_old[i]
            # Compute approximate relative error
            if x[i] != 0:
                ea[i] = abs((x[i] - x_old[i]) / x[i]) * 100

        iter_count += 1

        # Check stopping criteria
        if np.max(ea) < es or iter_count >= maxit:
            break

    return x, ea, iter_count


if __name__ == "__main__":

    # ### Problem 11.10: Condition Number of a Matrix ###
    # Generate the 10x10 Hilbert matrix
    H = hilbert(10)
    # Compute the condition number
    condition_number = cond(H)
    print(f"Condition number: {condition_number:.2e}")
    # Compute the right-hand side vector
    b = np.sum(H, axis=1)

    # Solve the system Hx = b
    x_computed = np.linalg.solve(H, b)

    # Calculate the error
    error = np.max(np.abs(x_computed - np.ones(10)))
    print(f"Maximum error: {error:.2e}")


    # ### Problem 11.11:  Condition Number of a Vandermonde Matrix ###
    # Define the x values
    x = np.array([4, 2, 7, 10, 3, 5])
    # Generate the Vandermonde matrix
    V = np.vander(x)
    # Compute the condition number
    condition_number_v = cond(V)
    print(f"Condition number of Vandermonde matrix: {condition_number_v:.2e}")
    # Compute the right-hand side vector
    b_v = V @ np.ones(6)

    # Solve the system Vx = b
    x_computed_v = np.linalg.solve(V, b_v)

    # Calculate the error
    error_v = np.max(np.abs(x_computed_v - np.ones(6)))
    print(f"Maximum error for Vandermonde system: {error_v:.2e}")



    # ### Problem 12.2: Gauss ###############
    A = np.array([[3, -0.1, -0.2],
                [0.1, 7, -0.3],
                [0.3, -0.2, 10]])
    b = np.array([7.85, -19.3, 71.4])

    # Solve using the Gauss-Seidel method with relaxation
    x, ea, iter_count = GaussSeidelR(A, b, lambda_relax=1.0, es=1e-4, maxit=50)

    print("Example 12.2 Results:")
    print(f"Solution (x): {x}")
    print(f"Approximate Errors (ea): {ea}")
    print(f"Number of Iterations: {iter_count}")

    # Solve Problem 12.2b using the same function
    A_12b = np.array([[4, -1, 0, 0, 0, 0],
                    [-1, 4, -1, 0, -1, 0],
                    [0, -1, 4, 0, 0, -1],
                    [0, 0, 0, 4, -1, 0],
                    [0, -1, 0, -1, 4, -1],
                    [0, 0, -1, 0, -1, 4]])
    b_12b = np.array([100, 100, 100, 100, 100, 100])

    x_12b, ea_12b, iter_count_12b = GaussSeidelR(A_12b, b_12b, lambda_relax=1.0, es=1e-4, maxit=50)

    print("\nProblem 12.2b Results:")
    print(f"Solution (x): {x_12b}")
    print(f"Approximate Errors (ea): {ea_12b}")
    print(f"Number of Iterations: {iter_count_12b}")


    # ### Problem 13.10: Eigenvalues and Eigenvectors ###
    # Problem parameters
    E = 10e9  # Modulus of elasticity (Pa)
    I = 1.25e-5  # Moment of inertia (m^4)
    L = 3.0  # Length of the column (m)
    P = 1e6  # Axial load (N)

    # Compute p^2
    p_squared = P / (E * I)

    # Number of interior nodes
    n = 4
    dx = L / (n + 1)  # Spacing between nodes

    # Construct the coefficient matrix A
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = -2 - dx**2 * p_squared
        if i > 0:
            A[i, i-1] = 1
        if i < n - 1:
            A[i, i+1] = 1

    print("Coefficient matrix A:")
    print(A)

    # Solve for eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eig(A)
    eigenvalues = np.real(eigenvalues)  # Keep only real parts
    eigenvectors = np.real(eigenvectors)  # Keep only real parts

    print("\nEigenvalues:")
    print(eigenvalues)

    print("\nEigenvectors:")
    print(eigenvectors)

    #####################################
    #Problem 13.11 (d)
    # Define the time vector
    t = np.linspace(0, 1, 1000)

    # Compute y1 and y2 based on the given equations
    y1 = 52.67 * 0.9477 * np.exp(-3.9899 * t) + 99.33 * (-0.0101) * np.exp(-302.0101 * t)
    y2 = 52.67 * 0.3191 * np.exp(-3.9899 * t) + 99.33 * 0.9999 * np.exp(-302.0101 * t)

    # Plot the results
    plt.figure()
    plt.plot(t, y1, 'b', linewidth=2, label=r'$y_1(t)$')
    plt.plot(t, y2, 'r', linewidth=2, label=r'$y_2(t)$')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$y(t)$')
    plt.legend()
    plt.title('Solution of the System of ODEs')
    plt.grid(True)
    plt.show()

    ###################################
    # Problem 2.4

    # Parameters
    Q_lung = 2  # μl/min
    Q_liver = 0.5  # μl/min
    Q_ot = 1.5  # μl/min
    V_lung = 0.08  # μl
    V_liver = 0.322  # μl
    v_max_P450_lung = 8.75  # μM/min
    v_max_P450_liver = 118  # μM/min
    v_max_EH_lung = 26.5  # μM/min
    K_m_EH_lung = 4.0  # μM
    v_max_EH_liver = 336  # μM/min
    K_m_EH_liver = 21  # μM
    v_max_GST_lung = 2750  # μM/min
    K1_lung = 310000  # μM²
    K2_lung = 35  # μM
    v_max_GST_liver = 2750  # μM/min
    K1_liver = 150000  # μM²
    K2_liver = 35  # μM
    k_NOH = 0.173  # μM/μM of NO/min
    k_OH = -20.2  # ml/g protein
    TP_lung = 92  # mg/ml
    TP_liver = 192  # mg/ml
    C_GSH_lung = 1800  # μM
    C_GSH_liver = 7500  # μM

    # Recycle fraction range
    R_values = np.arange(0.6, 0.95, 0.05)

    # Initialize arrays to store results
    C_NO_lung_values = np.zeros(len(R_values))
    C_NO_liver_values = np.zeros(len(R_values))

    # Loop over R values
    for i, R in enumerate(R_values):
        # Matrix elements
        A11 = -Q_lung - (v_max_EH_lung * V_lung) / K_m_EH_lung - \
            V_lung * (v_max_GST_lung * C_GSH_lung) / (K1_lung + K2_lung * C_GSH_lung) - \
            k_NOH * np.exp(k_OH * TP_lung) * V_lung
        A12 = R * Q_liver
        A21 = Q_liver
        A22 = -Q_liver - (v_max_EH_liver * V_liver) / K_m_EH_liver - \
            V_liver * (v_max_GST_liver * C_GSH_liver) / (K1_liver + K2_liver * C_GSH_liver) - \
            k_NOH * np.exp(k_OH * TP_liver) * V_liver
        B1 = -R * Q_ot * C_NO_lung_values[i] - v_max_P450_lung * V_lung
        B2 = -v_max_P450_liver * V_liver
        
        # Gaussian elimination
        # Forward elimination
        factor = A21 / A11
        A22 = A22 - factor * A12
        B2 = B2 - factor * B1
        
        # Back substitution
        C_NO_liver = B2 / A22
        C_NO_lung = (B1 - A12 * C_NO_liver) / A11
        
        # Store results
        C_NO_lung_values[i] = C_NO_lung
        C_NO_liver_values[i] = C_NO_liver

    # Plot results
    plt.plot(R_values, C_NO_lung_values, 'b-o', label=r'$C^{NO}_{lung}$')
    plt.plot(R_values, C_NO_liver_values, 'r-x', label=r'$C^{NO}_{liver}$')
    plt.xlabel('Recycle fraction R')
    plt.ylabel('Concentration (μM)')
    plt.legend()
    plt.title('Concentration of Naphthalene Epoxide in Lung and Liver')
    plt.grid(True)
    plt.show()

    # LU decomposition verification
    for i, R in enumerate(R_values):
        # Matrix elements
        A = np.array([[A11, A12], [A21, A22]])
        B = np.array([B1, B2])
        
        # LU decomposition
        P, L, U = lu(A)
        
        # Forward substitution (Ly = B)
        y = np.linalg.solve(L, B)
        
        # Back substitution (Ux = y)
        x = np.linalg.solve(U, y)
        
        # Verify results
        print(f'R = {R:.2f}: C_NO_lung = {x[0]:.4f}, C_NO_liver = {x[1]:.4f}')

