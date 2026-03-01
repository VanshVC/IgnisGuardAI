import numpy as np
import matplotlib.pyplot as plt

def plot_spectral_signature(bands, labels=['B2', 'B3', 'B4', 'B8', 'B11', 'B12']):
    """
    Visualizes the spectral signature of a pixel to identify fire.
    Fire shows high reflectance in SWIR (B11, B12) and low in NIR (B8) 
    compared to healthy vegetation.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(labels, bands, marker='o', linestyle='-', color='red', label='Fire Signal')
    
    # Typical vegetation baseline
    veg_baseline = [0.02, 0.04, 0.03, 0.45, 0.2, 0.1]
    plt.plot(labels, veg_baseline, marker='x', linestyle='--', color='green', label='Healthy Forest')
    
    plt.title("Spectral Signature Analysis: Fire vs Vegetation")
    plt.xlabel("Spectral Bands (Wavelength)")
    plt.ylabel("Reflectance")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('data/spectral_analysis.png')
    print("Spectral analysis plot saved to data/spectral_analysis.png")

if __name__ == "__main__":
    # Example values for a 'Fire' pixel in Sentinel-2
    # High SWIR (B12) is the primary indicator
    fire_pixel = [0.05, 0.08, 0.12, 0.35, 0.8, 0.9]
    plot_spectral_signature(fire_pixel)