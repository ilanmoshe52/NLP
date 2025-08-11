import numpy as np

# Prceise location convergence using Newton Rapson (for 3 GPS points).
#
# #Parameters:
#- gps_points: List of 3 GPS points (x, y, z) in meters.
#- distances: List of distances to each GPS point.
#- initial_guess: Initial guess for (x, y, z).
#- max_iters: Maximum number of iterations.
#- tol: Convergence tolerance.
#
#Returns:
#- Estimated position (x, y, z).
def location_convergence(gps_points, distances, initial_guess, max_iters=100, tol=1e-6):

    if len(gps_points) != 3:
        raise ValueError("This method requires 3 GPS points.")

    # Initialize the estimated position
    x_est = np.array(initial_guess, dtype=np.float64)

    for iteration in range(max_iters):
        # Construct the system of equations
        A = []
        b = []

        for i, (P, d) in enumerate(zip(gps_points, distances)):
            x_i, y_i, z_i = P
            A.append([
                2 * (x_est[0] - x_i),
                2 * (x_est[1] - y_i),
                2 * (x_est[2] - z_i)
            ])
            b.append([
                (x_est[0] - x_i)**2 + (x_est[1] - y_i)**2 + (x_est[2] - z_i)**2 - d**2
            ])

        A = np.array(A)
        b = np.array(b)

        # Solve directly using matrix inverse
        try:
            # inverse Jacobian multiplies by error vactor
            delta = np.linalg.inv(A) @ (-b)
        except np.linalg.LinAlgError:
            print("Matrix is singular and cannot be inverted.")
            return x_est

        # Update the estimate
        x_est = x_est + delta.flatten()

        # Check for convergence
        if np.linalg.norm(delta) < tol:
            print(f"Converged after {iteration + 1} iterations.")
            return x_est

    print("Did not converge within the maximum number of iterations.")
    return x_est


# Example 
gps_points = [(0, 0, 1000), (1000, 0, 1200), (500, 800, 1100)]
distances = [950, 1200, 850]
initial_guess = [500, 500, 500]

estimated_position = location_convergence(gps_points, distances, initial_guess)
print("Initial Position:", initial_guess)
print("Estimated Position:", estimated_position)