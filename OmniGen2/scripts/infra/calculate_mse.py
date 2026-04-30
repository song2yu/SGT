from PIL import Image
import numpy as np

def calculate_mse(image_path_1, image_path_2):
    """
    Compute the mean squared error (MSE) between two images.
    
    Args:
    image_path_1 (str): path to the first image.
    image_path_2 (str): path to the second image.
    
    Returns:
    float: the MSE between the two images.
    """
    try:
        # Load the image and convert it to a NumPy array
        img1 = np.array(Image.open(image_path_1).convert('L'))  # .convert('L') Convert to grayscale
        img2 = np.array(Image.open(image_path_2).convert('L'))
    except FileNotFoundError:
        print("Error: file not found, please verify the path.")
        return None

    # Make sure the two images share the same shape
    if img1.shape != img2.shape:
        print("Error: image shapes do not match.")
        return None

    # Compute the mean squared error (MSE)
    # Cast to float to avoid overflow when computing the difference
    mse_value = np.mean((img1.astype("float") - img2.astype("float")) ** 2)
    
    return mse_value

# Path to the image file
image1_path = "example_images/289089159-a6d7abc142419e63cab0a566eb38e0fb6acb217b340f054c6172139b316f6596_complex64.png"
image2_path = "example_images/289089159-a6d7abc142419e63cab0a566eb38e0fb6acb217b340f054c6172139b316f6596_complex128.png"

# Compute and print the MSE
mse = calculate_mse(image1_path, image2_path)
if mse is not None:
    print(f"MSE between the two images is: {mse:.2f}")