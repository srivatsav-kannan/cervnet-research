import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from skimage.util import random_noise


def compute_kurtosis(G, mu):
    """Compute the kurtosis value from noise intensity G."""
    N = len(G)
    numerator = np.sum((G - mu) ** 4) / N
    denominator = (np.sum((G - mu) ** 2) / N) ** 2
    return numerator / denominator - 3


def compute_correlation(image, noise):
    """Compute the correlation between image and noise intensities."""
    Ix, Iy = np.gradient(image)
    Gx, Gy = np.gradient(noise)

    num_I = np.sum(Ix * Iy) - np.mean(Ix) * np.mean(Iy)
    den_I = np.sum((Ix - np.mean(Ix)) * (Iy - np.mean(Iy)))

    num_G = np.sum(Gx * Gy) - np.mean(Gx) * np.mean(Gy)
    den_G = np.sum((Gx - np.mean(Gx)) * (Gy - np.mean(Gy)))

    rho_I = num_I / (den_I + 1e-8)
    rho_G = num_G / (den_G + 1e-8)

    return rho_I, rho_G


def modified_anisotropic_diffusion(image, iterations=50, k_threshold=0.001):
    """Apply Modified Anisotropic Diffusion Filtering (MADF)."""
    image = image.astype(np.float32) / 255.0  # Normalize image
    noisy_image = random_noise(image, mode='speckle')  # Generate speckle noise
    noise = noisy_image - image
    G = gaussian_filter(noise, sigma=1)  # Estimate noise intensity
    mu = np.mean(G)

    for i in range(iterations):
        k = compute_kurtosis(G, mu)
        rho_I, rho_G = compute_correlation(image, noise)

        if abs(k) <= k_threshold:
            break  # Stop iterations if kurtosis is below threshold

        # Compute gradients
        Ix, Iy = np.gradient(image)
        gradient_magnitude = np.sqrt(Ix ** 2 + Iy ** 2)

        # Diffusion function based on gradient magnitude
        diffusion_coefficient = np.exp(- (gradient_magnitude ** 2) / (2 * np.var(gradient_magnitude)))

        # Apply diffusion process
        image = image + diffusion_coefficient * gaussian_filter(image, sigma=1)

        # Update noise estimation
        noise = random_noise(image, mode='speckle') - image
        G = gaussian_filter(noise, sigma=1)
        mu = np.mean(G)

    return (image * 255).astype(np.uint8)  # Convert back to uint8


# Load an example image
input_image = cv2.imread("input_image.png", cv2.IMREAD_GRAYSCALE)  # Load in grayscale
filtered_image = modified_anisotropic_diffusion(input_image)

# Save and display the result
cv2.imwrite("filtered_image.png", filtered_image)
cv2.imshow("Filtered Image", filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
