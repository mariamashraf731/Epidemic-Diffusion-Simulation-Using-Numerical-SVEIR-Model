import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.interpolate import griddata



if __name__ == "__main__":


    # ### Problem 15.19
    # Given data
    T_a = np.array([273, 283, 293, 303, 313])  # Absolute temperature (K)
    K_w = np.array([1.164e-15, 2.950e-15, 6.846e-15, 1.467e-14, 2.292e-14])  # Ion product of water

    # Transform K_w to y = -log10(K_w)
    y = -np.log10(K_w)

    # Define the regression function
    def regression_func(T_a, a, b, c, d):
        return a / T_a + b * np.log10(T_a) + c * T_a + d

    # Fit the model to the data
    params, covariance = curve_fit(regression_func, T_a, y)

    # Extract the parameters
    a, b, c, d = params

    # Predicted values
    y_pred = regression_func(T_a, a, b, c, d)

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.scatter(T_a, y, color='red', label='Given Data (-log10 K_w)')
    plt.plot(T_a, y_pred, label='Fitted Model', color='blue')
    plt.xlabel('T_a (K)')
    plt.ylabel('-log10(K_w)')
    plt.title('Regression Fit for Ion Product of Water')
    plt.legend()
    plt.grid()
    plt.show()

    # Print the results
    print("Estimated Parameters:")
    print(f"a = {a:.4e}")
    print(f"b = {b:.4e}")
    print(f"c = {c:.4e}")
    print(f"d = {d:.4e}")

    print("\nPredicted Values (-log10 K_w):")
    for T, y_val in zip(T_a, y_pred):
        print(f"T_a = {T} K: -log10(K_w) = {y_val:.4f}")
    
    ######################################
    #Problem 18.3
    # Given data
    x = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    f_x = np.array([5.176, 15.471, 45.887, 96.500, 47.448, 19.000, 11.692, 12.382, 17.846, 21.703, 16.000])

    # Define the exact humps function
    def humps(x):
        return 1 / ((x - 0.3)**2 + 0.01) + 1 / ((x - 0.9)**2 + 0.04) - 6

    # Generate a finer grid for plotting
    x_fine = np.linspace(0, 1, 500)
    f_exact = humps(x_fine)

    # (a) Cubic spline with not-a-knot end conditions
    cs = CubicSpline(x, f_x, bc_type='not-a-knot')
    f_cs = cs(x_fine)

    # (b) Piecewise cubic Hermite interpolation
    pchip = PchipInterpolator(x, f_x)
    f_pchip = pchip(x_fine)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x_fine, f_exact, label='Exact Humps Function', color='black', linestyle='--')
    plt.plot(x_fine, f_cs, label='Cubic Spline (Not-a-Knot)', color='blue')
    plt.plot(x_fine, f_pchip, label='Piecewise Cubic Hermite', color='red')
    plt.scatter(x, f_x, color='green', label='Data Points', zorder=5)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Comparison of Interpolation Methods with Exact Humps Function')
    plt.legend()
    plt.grid(True)
    plt.show()

    #####################################
    #Problem 18.12

    # Step 1: Generate the data points
    t = np.linspace(0, 2*np.pi, 8)  # Eight points from 0 to 2*pi
    f = np.sin(t)**2                # f(t) = sin^2(t)

    # True function for comparison
    t_true = np.linspace(0, 2*np.pi, 1000)  # Dense grid for true function
    f_true = np.sin(t_true)**2                # True function values

    # Step 2: Fit using cubic spline with Not-a-Knot end conditions
    cs_not_a_knot = CubicSpline(t, f, bc_type='not-a-knot')
    spline_not_a_knot = cs_not_a_knot(t_true)

    # Step 3: Fit using cubic spline with derivative end conditions
    # Derivatives of the true function
    f_prime = 2 * np.sin(t) * np.cos(t)   # Derivative of sin^2(t)

    # Cubic spline with derivative conditions
    cs_derivative = CubicSpline(t, f, bc_type=((1, f_prime[0]), (1, f_prime[-1])))
    spline_derivative = cs_derivative(t_true)

    # Step 4: Fit using Piecewise Cubic Hermite Interpolation (PCHIP)
    pchip_fit = PchipInterpolator(t, f)(t_true)

    # Step 5: Plot the results
    plt.figure(figsize=(10, 6))

    # Plot the true function
    plt.plot(t_true, f_true, 'k', label='True function', linewidth=2)

    # Plot cubic spline with Not-a-Knot conditions
    plt.plot(t_true, spline_not_a_knot, 'r--', label='Cubic Spline (Not-a-Knot)', linewidth=2)

    # Plot cubic spline with derivative conditions
    plt.plot(t_true, spline_derivative, 'b-.', label='Cubic Spline (Derivative)', linewidth=2)

    # Plot Piecewise Cubic Hermite Interpolation
    plt.plot(t_true, pchip_fit, 'g:', label='Piecewise Cubic Hermite Interpolation', linewidth=2)

    # Labeling the plot
    plt.legend()
    plt.title('Spline Interpolation Fits')
    plt.xlabel('t')
    plt.ylabel('f(t)')
    plt.grid(True)
    plt.show()

    # Step 6: Calculate the absolute error for each method
    error_not_a_knot = np.abs(spline_not_a_knot - f_true)
    error_derivative = np.abs(spline_derivative - f_true)
    error_pchip = np.abs(pchip_fit - f_true)

    # Plot the absolute errors
    plt.figure(figsize=(10, 6))
    plt.plot(t_true, error_not_a_knot, 'r--', label='Error (Not-a-Knot)', linewidth=2)
    plt.plot(t_true, error_derivative, 'b-.', label='Error (Derivative)', linewidth=2)
    plt.plot(t_true, error_pchip, 'g:', label='Error (PCHIP)', linewidth=2)

    # Labeling the error plot
    plt.legend()
    plt.title('Absolute Errors for Each Method')
    plt.xlabel('t')
    plt.ylabel('Absolute Error')
    plt.grid(True)
    plt.show()

    ###################################
    # Problem 18.14

    # Define the temperature distribution function T(x, y)
    def temperature(x, y):
        return 2 + x - y + 2*x**2 + 2*x*y + y**2

    # (a) Generate a meshgrid for x and y
    x = np.linspace(-2, 0, 100)
    y = np.linspace(0, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = temperature(X, Y)

    # Plot the meshplot using matplotlib
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Temperature (T)')
    ax.set_title('Temperature Distribution on the Rectangular Plate')
    plt.show()
    # (b) Define query point
    x_query = -1.63
    y_query = 1.627

    # (b) Use griddata (linear interpolation) to compute the temperature at x = -1.63 and y = 1.627
    x_vals = np.linspace(-2, 0, 9)
    y_vals = np.linspace(0, 3, 9)
    X_vals, Y_vals = np.meshgrid(x_vals, y_vals)
    Z_vals = temperature(X_vals, Y_vals)

    # Perform linear interpolation using griddata
    points = np.array([X_vals.flatten(), Y_vals.flatten()]).T
    values = Z_vals.flatten()
    temperature_bilinear = griddata(points, values, (x_query, y_query), method='linear')

    # True value calculation (for error calculation)
    x_query = -1.63
    y_query = 1.627
    true_temperature = temperature(x_query, y_query)

    # Calculate the percent relative error for bilinear interpolation
    error_bilinear = abs((temperature_bilinear - true_temperature) / true_temperature) * 100

    print(f'Bilinear Interpolation - Temperature at x = {x_query}, y = {y_query}: {temperature_bilinear}')
    print(f'Percent Relative Error for Bilinear Interpolation: {error_bilinear}%')

    # (c) Use spline interpolation using griddata with cubic method
    temperature_spline = griddata(points, values, (x_query, y_query), method='cubic')

    # Calculate the percent relative error for spline interpolation
    error_spline = abs((temperature_spline - true_temperature) / true_temperature) * 100

    print(f'Spline Interpolation - Temperature at x = {x_query}, y = {y_query}: {temperature_spline}')
    print(f'Percent Relative Error for Spline Interpolation: {error_spline}%')
