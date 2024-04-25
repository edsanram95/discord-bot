import numpy as np
import matplotlib.pyplot as plt

# Define the CDF function
def cdf(x):
    return 1 - x**(-4)

# Generate x values
x_values = np.linspace(1.01, 10, 100)  # Start from a bit above 1 to avoid division by zero

# Calculate the corresponding y values using the CDF function
y_values = cdf(x_values)

# Plot the CDF
plt.plot(x_values, y_values, label='CDF')

# Add labels and title
plt.xlabel('x')
plt.ylabel('F(x)')
plt.title('Cumulative Distribution Function (CDF) of X')

# Add a legend
plt.legend()

# Display the plot
plt.grid(True)
plt.show()
