import numpy as np
import matplotlib.pyplot as plt
from math import factorial




if __name__ == "__main__":

    # Problem 2.5
    # Define constants
    a = 2
    b = 5
    x = np.arange(0, np.pi/2 + np.pi/40, np.pi/40)  # Vector x from 0 to π/2 with step π/40

    # Compute y using the given formula
    y = b * np.exp(-a * x) * np.sin(b * x) * (0.012 * x**4 - 0.15 * x**3 + 0.075 * x**2 + 2.5 * x)

    # Compute z as the square of each element in y
    z = y**2

    # Combine x, y, and z into a matrix w
    w = np.column_stack((x, y, z))

    # Display the matrix w
    np.set_printoptions(precision=4, suppress=True)  # Short 'g' format
    print("Matrix w:")
    print(w)

    # Plot y and z versus x
    plt.figure(figsize=(10, 6))

    # Plot y
    plt.plot(x, y, '-.r', linewidth=1.5, marker='p', markerfacecolor='white', markeredgecolor='red', markersize=14, label='y')

    # Plot z
    plt.plot(x, z, '-b', marker='s', markerfacecolor='green', markeredgecolor='blue', label='z')

    # Add labels, legend, and grid
    plt.xlabel('x', fontsize=12)
    plt.ylabel('Values of y and z', fontsize=12)
    plt.title('y and z vs x', fontsize=14)
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()
    #####################################
    # Problem 2.10 
    # Define the parameter matrix
    data = np.array([
        [0.035, 0.0001, 10, 2],
        [0.020, 0.0002, 8, 1],
        [0.015, 0.0010, 20, 1.5],
        [0.030, 0.0007, 24, 3],
        [0.022, 0.0003, 15, 2.5]
    ])

    # Extract individual columns for computation
    n = data[:, 0]  # Roughness coefficient
    S = data[:, 1]  # Channel slope
    B = data[:, 2]  # Width
    H = data[:, 3]  # Depth

    # Compute the velocity U using Manning's equation
    U = (np.sqrt(S) / n) * ((B * H) / (B + 2 * H)) ** (2/3)

    # Display the computed velocities
    print("Computed Velocities (m/s):")
    print(U)
    #############################
    #Problem 2.12

    # Given data
    t_data = np.array([10, 20, 30, 40, 50, 60])  # Time (min)
    c_data = np.array([3.4, 2.6, 1.6, 1.3, 1.0, 0.5])  # Concentration (ppm)

    # Function for concentration
    def concentration(t):
        return 4.84 * np.exp(-0.034 * t)

    # Time range for the function
    t_func = np.linspace(0, 70, 500)  # From 0 to 70 min with finer granularity
    c_func = concentration(t_func)  # Compute function values

    # Plot the data points
    plt.scatter(t_data, c_data, color='red', marker='D', label='Data (Measured)', zorder=5)

    # Plot the function
    plt.plot(t_func, c_func, 'g--', label='Function (c = 4.84e^{-0.034t})')

    # Add labels and legend
    plt.xlabel('Time (min)')
    plt.ylabel('Concentration (ppm)')
    plt.title('Photodegradation of Aqueous Bromine')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

    # Given data
    t_data = np.array([10, 20, 30, 40, 50, 60])  # Time (min)
    c_data = np.array([3.4, 2.6, 1.6, 1.3, 1.0, 0.5])  # Concentration (ppm)

    # Function for concentration
    def concentration(t):
        return 4.84 * np.exp(-0.034 * t)

    # Time range for the function
    t_func = np.linspace(0, 70, 500)  # From 0 to 70 min with finer granularity
    c_func = concentration(t_func)  # Compute function values

    # Plot the data and function using a logarithmic y-axis
    plt.semilogy(t_data, c_data, 'rD', label='Data (Measured)', zorder=5)  # Discrete data as red diamonds
    plt.semilogy(t_func, c_func, 'g--', label='Function (c = 4.84e^{-0.034t})')  # Function as a green dashed line

    # Add labels, title, and legend
    plt.xlabel('Time (min)')
    plt.ylabel('Concentration (ppm) [Log Scale]')
    plt.title('Photodegradation of Aqueous Bromine (Log Scale)')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()
    """
    # Explanation of Results

    ## 1. Semilogarithmic Plot
    - The `semilogy` function plots the x-axis on a linear scale and the y-axis on a logarithmic (base-10) scale.
    - This transforms exponential decay into a straight line, simplifying the visualization and analysis of data that spans multiple orders of magnitude.

    ## 2. Interpretation
    - The discrete data points (red diamonds) align well with the green dashed line representing the function \( c(t) = 4.84e^{-0.034t} \), confirming the accuracy of the model.
    - On a logarithmic scale, exponential decay becomes linear, and deviations (if any) between the model and data are more apparent.

    ## 3. Why Use `semilogy`?
    - Logarithmic scaling highlights small values, improving readability when values differ by orders of magnitude.
    - It is particularly useful for analyzing processes like photodegradation, which follow an exponential trend.

    ## 4. Observations
    - The straight-line behavior of the function on the semilogarithmic plot validates the exponential decay model.
    - Any significant deviation of the data points from the line would indicate noise, measurement errors, or an imperfect model fit.
    """
    #####################################
    #Problem 2.13

    # Given data
    v_data = np.array([10, 20, 30, 40, 50, 60, 70, 80])  # Velocity (m/s)
    F_data = np.array([25, 70, 380, 550, 610, 1220, 830, 1450])  # Force (N)

    # Function for force
    def force(v):
        return 0.2741 * v**1.9842

    # Velocity range for the function
    v_func = np.linspace(0, 100, 500)  # From 0 to 100 m/s
    F_func = force(v_func)  # Compute function values

    # Plot the data and the function
    plt.figure(figsize=(8, 6))
    plt.plot(v_func, F_func, 'k-.', label='F = 0.2741v^{1.9842}')  # Function (black dash-dotted line)
    plt.scatter(v_data, F_data, color='magenta', label='Data (Measured)', zorder=5)  # Data (circular magenta symbols)

    # Plot labels and title
    plt.xlabel('Velocity (v) [m/s]', fontsize=12)
    plt.ylabel('Force (F) [N]', fontsize=12)
    plt.title('Force vs Velocity (Linear Scale)', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()
    ###############################
    #Problem 2.14
    # Plot the data and the function on a logarithmic scale
    plt.figure(figsize=(8, 6))
    plt.loglog(v_func, F_func, 'k-.', label='F = 0.2741v^{1.9842}')  # Function (log-log, black dash-dotted line)
    plt.scatter(v_data, F_data, color='magenta', label='Data (Measured)', zorder=5)  # Data (circular magenta symbols)

    # Plot labels and title
    plt.xlabel('Velocity (v) [m/s]', fontsize=12)
    plt.ylabel('Force (F) [N]', fontsize=12)
    plt.title('Force vs Velocity (Logarithmic Scale)', fontsize=14)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()
    """
    # Explanation for Logarithmic Plot (`loglog`)
    - The `loglog` function transforms both the x-axis and y-axis to a logarithmic (base-10) scale.
    - In this case, the relationship \( F = 0.2741v^{1.9842} \) becomes a straight line on the log-log plot because the equation represents a power-law relationship.
    - The slope of this line corresponds to the exponent (1.9842) in the power-law equation.
    - **Logarithmic scaling** is helpful to:
    - Highlight multiplicative or exponential relationships.
    - Identify deviations from the expected model (if data points deviate from the straight line).
    - **Observations**:
    - The measured data aligns closely with the theoretical model, confirming the power-law relationship.
    - Deviations might indicate noise or non-ideal conditions during measurements.
    """
    #############################
    # Problem 2.15

    # Define the Maclaurin series for cosine
    def maclaurin_cos(x, terms=5):
        """
        Compute the cosine of x using the Maclaurin series expansion up to 'terms'.
        """
        cos_x = 0
        for n in range(terms):
            cos_x += ((-1)**n) * (x**(2*n)) / factorial(2*n)
        return cos_x

    # Define the range of x values
    x = np.linspace(0, 3 * np.pi / 2, 500)

    # Compute the actual cosine values
    cos_values = np.cos(x)

    # Compute the Maclaurin series approximation
    maclaurin_values = maclaurin_cos(x, terms=5)  # Up to x^8 / 8!

    # Plot the actual cosine function
    plt.plot(x, cos_values, label='cos(x)', color='blue', linestyle='-')

    # Plot the Maclaurin series approximation
    plt.plot(x, maclaurin_values, label='Maclaurin Approximation', color='black', linestyle='--')

    # Add labels, legend, and title
    plt.xlabel('x')
    plt.ylabel('cos(x)')
    plt.title('Cosine Function vs. Maclaurin Series Approximation')
    plt.legend()
    plt.grid()

    # Show the plot
    plt.show()
    #######################################
    # Problem 2.19: Projectile Motion
    # Constants
    g = 9.81  # gravitational acceleration (m/s^2)
    v0 = 28  # initial velocity (m/s)
    y0 = 0  # initial height (m)
    angles_deg = np.arange(15, 76, 15)  # angles in degrees
    angles_rad = np.radians(angles_deg)  # convert to radians
    x_values = np.arange(0, 81, 5)  # horizontal distances (m)

    # Matrix to store heights
    trajectory_matrix = np.zeros((len(x_values), len(angles_deg)))

    # Calculate the trajectory for each angle
    for i, theta in enumerate(angles_rad):
        for j, x in enumerate(x_values):
            # Calculate height
            y = (np.tan(theta) * x) - (g * x**2) / (2 * (v0**2) * (np.cos(theta)**2)) + y0
            trajectory_matrix[j, i] = y if y >= 0 else 0  # Ensure height is non-negative

    # Plot trajectories
    plt.figure(figsize=(10, 6))
    for i, theta in enumerate(angles_deg):
        plt.plot(x_values, trajectory_matrix[:, i], label=f"{theta}°")

    # Configure plot
    plt.title("Trajectories of the Object")
    plt.xlabel("Horizontal Distance (m)")
    plt.ylabel("Height (m)")
    plt.legend(title="Initial Angle")
    plt.axis([0, 80, 0, np.max(trajectory_matrix) + 5])  # Adjust axes to ensure min height is 0
    plt.grid(True)
    plt.show()


    # Problem 2.20: Reaction Rate
    # Constants
    A = 7e16  # Pre-exponential factor (s^-1)
    E = 1e5  # Activation energy (J/mol)
    R = 8.314  # Gas constant (J/(mol*K))
    T_range = np.arange(253, 326, 1)  # Temperature range (K)

    # Calculate reaction rates
    k = A * np.exp(-E / (R * T_range))  # Reaction rate
    inverse_T = 1 / T_range  # Inverse temperature (1/K)

    # Create plots
    plt.figure(figsize=(12, 6))

    # Subplot (a): k vs Ta (green line)
    plt.subplot(1, 2, 1)
    plt.plot(T_range, k, 'g-', label='k vs Ta')
    plt.xlabel("Temperature (K)")
    plt.ylabel("Reaction Rate (k) [s⁻¹]")
    plt.title("Reaction Rate vs Temperature")
    plt.grid(True)

    # Subplot (b): k vs 1/Ta using semilogy (red line)
    plt.subplot(1, 2, 2)
    plt.semilogy(inverse_T, k, 'r-', label='semilogy(k) vs 1/Ta')
    plt.xlabel("1/Temperature (1/K)")
    plt.ylabel("Reaction Rate (k) [s⁻¹] (log scale)")
    plt.title("Reaction Rate vs Inverse Temperature (Semilogy)")
    plt.grid(True)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()



    # Problem 2.22 : Parametric Butterfly Curve (x, y)
    # Define the parametric equations
    t = np.linspace(0, 100, 1601)  # t from 0 to 100, Δt = 1/16
    x = np.sin(t) * (np.exp(np.cos(t)) - 2 * np.cos(4 * t) - (np.sin(t / 12)**5))
    y = np.cos(t) * (np.exp(np.cos(t)) - 2 * np.cos(4 * t) - (np.sin(t / 12)**5))

    # Create plots
    plt.figure(figsize=(10, 10))

    # Subplot (a): x and y vs t
    plt.subplot(2, 1, 1)
    plt.plot(t, x, 'b-', label='x(t)', linewidth=1)
    plt.plot(t, y, 'r--', label='y(t)', linewidth=1)
    plt.title("Butterfly Curve: x(t) and y(t)")
    plt.xlabel("t")
    plt.ylabel("Values of x and y")
    plt.legend()
    plt.grid(True)

    # Subplot (b): y vs x
    plt.subplot(2, 1, 2)
    plt.plot(x, y, 'g-', linewidth=1)
    plt.title("Butterfly Curve: y vs x")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('square')  # Make the plot square
    plt.grid(True)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()



    # Problem 2.23: Polar Representation of the Butterfly Curve
    # Define the polar equation
    theta = np.linspace(0, 8 * np.pi, 256)  # θ from 0 to 8π, Δθ = π/32
    r = np.exp(np.sin(theta)) - 2 * np.cos(4 * theta) - (np.sin((2 * theta - np.pi) / 24)**5)

    # Create polar plot
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)  # Create polar subplot
    ax.plot(theta, r, 'r--', linewidth=1)  # Dashed red line

    # Add title and show the plot
    ax.set_title("Butterfly Curve in Polar Coordinates", va='bottom')
    plt.show()




    # Problem 3.6
    def cartesian_to_polar(x, y):
        """
        Converts Cartesian coordinates (x, y) to polar coordinates (r, θ).
        Returns r (radius) and θ (angle in radians).
        """
        r = np.sqrt(x**2 + y**2)  # Compute radius
        theta = np.arctan2(y, x)  # Compute angle (handles all quadrants)
        return r, theta

    # Example: Test the function with Cartesian coordinates
    x_values = [1, -1, -1, 1]
    y_values = [1, 1, -1, -1]

    for x, y in zip(x_values, y_values):
        r, theta = cartesian_to_polar(x, y)
        print(f"Cartesian: (x={x}, y={y}) -> Polar: (r={r:.2f}, θ={theta:.2f} radians)")

