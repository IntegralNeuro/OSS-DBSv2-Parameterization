import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

nii_segmask_values = {
    "none": 0,
    "white_matter": 1,
    "gray_matter": 2,
    "csf": 3
}


def load_nii(file_path):
    """
    Load a NIfTI file and return the image data and header information.
    
    Parameters:
    file_path (str): The path to the .nii file.
    
    Returns:
    Tuple[numpy.ndarray, nibabel.Nifti1Header]: The image data and header.
    """
    # Load the NIfTI file
    nii_img = nib.load(file_path)
    
    # Get the image data as a numpy array
    img_data = nii_img.get_fdata()
    
    # Get the header information
    header = nii_img.header
    
    return img_data, header


def save_nii(img_data, header, file_path):
    """
    Save the image data to a NIfTI file.
    
    Parameters:
    img_data (numpy.ndarray): The image data.
    header (nibabel.Nifti1Header): The header information.
    file_path (str): The path to save the .nii file.
    """
    nii_img = nib.Nifti1Image(img_data, None, header)
    nib.save(nii_img, file_path)


def plot_slice(img_data, slice_index):
    """
    Plot a single slice of the image data.
    
    Parameters:
    img_data (numpy.ndarray): The image data.
    slice_index (int): The index of the slice to plot.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(img_data[:, :, slice_index], cmap='gray')
    plt.title(f'Slice {slice_index}')
    plt.axis('off')
    plt.show()

def plot_3d(img_data):
    """
    Plot a 3D visualization of the image data using PyVista.
    
    Parameters:
    img_data (numpy.ndarray): The image data.
    """
    img_data = np.where(img_data == 0, np.nan, img_data)
    # Create a PyVista grid from the image data
    grid = pv.ImageData(dimensions=img_data.shape)
    grid.spacing = (1, 1, 1)  # Spacing between grid points
    grid.origin = (0, 0, 0)  # Origin of the grid
    grid.point_data['values'] = img_data.flatten(order='F')  # Flatten the image data

    # Set up the plotter
    p = pv.Plotter()
    opacity = [0, 0, 0, 0.1, 0.3, 0.6, 0.9, 1]
    # Add the volume with improved opacity settings
    p.add_volume(grid, scalars='values', cmap='gray', opacity=opacity)

    p.set_background('slategray')
    # Show the plotter
    p.show()


def make_homogeneous_segmask(segmask_filename, new_type):
    """
    Replace non-zero values in a segmask with a specified value and save the new segmask.
    
    Parameters:
    segmask_filename (str): The filename of the segmask.
    new_type (string): The tissue type replace non-zero values (see options below)

    Segmask tissue type strings:
    - "none"
    - "white_matter"
    - "gray_matter"
    - "csf"
    """
    # Load the segmask
    segmask = nib.load(segmask_filename)
    segmask_data = segmask.get_fdata()
    
    # Replace non-zero values with the specified value
    segmask_data[segmask_data != 0] = nii_segmask_values[new_type]
    
    # Get the segmask type from the filename
    segmask_type = segmask_filename.split(".")[0]
    
    # Save the new segmask with the type in the filename
    new_filename = f"{segmask_type}_{new_type}.nii"
    new_segmask = nib.Nifti1Image(segmask_data, segmask.affine, segmask.header)
    nib.save(new_segmask, new_filename)


# Example usage
if __name__ == "__main__":
    file_path = "segmask.nii"
    img_data, header = load_nii(file_path)
    
    # Plot a slice (e.g., the middle slice)
    # slice_index = img_data.shape[2] // 2
    # plot_slice(img_data, slice_index)
    
    # Plot a 3D visualization with slider
    plot_3d(img_data)
