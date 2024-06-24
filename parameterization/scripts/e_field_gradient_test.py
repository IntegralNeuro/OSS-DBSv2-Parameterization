import pyvista as pv
import numpy as np
import pandas as pd

# Load the mesh with E-field data
mesh = pv.read('./results_example/E-field.vtu')

# Compute the gradient of the electric field
gradient = mesh.compute_derivative(scalars='E_field_real', gradient=True)

# # Add the gradient to the mesh
# mesh.point_data['E_field_gradient'] = gradient

# # Create a plotter object
# plotter = pv.Plotter()

# # Create glyphs for the electric field with a very small factor to reduce the size of the arrows
# glyphs = mesh.glyph(orient='E_field', scale='E_field', factor=0.01, geom=pv.Arrow())  # Adjust factor to make arrows small

# # Add the glyphs to the plotter
# plotter.add_mesh(glyphs, color='red')

# # Convert gradient.points to a DataFrame
df = pd.DataFrame(gradient["gradient"], columns=['dxx', 'dxy', 'dxz', "dyx", "dyy", "dyz", "dzx", "dzy", "dzz"])

# # Save the DataFrame to a CSV file
df.to_csv('./results_example/E_field_gradient.csv', index=True)

# # Add the gradient to the mesh


# # Define the activation threshold (example value)
# activation_threshold = 0.2  # V/mm

# # Identify points where E-field exceeds the threshold
# activated_points = np.linalg.norm(mesh['E_field'], axis=1) > activation_threshold

# # Extract the activated volume
# activated_volume = mesh.extract_points(activated_points)

# # Visualize the activated volume
# plotter = pv.Plotter()
# plotter.add_mesh(activated_volume, color='red', opacity=0.5)
# plotter.add_mesh(mesh, scalars='E_field_gradient', vector=True)
# plotter.show()