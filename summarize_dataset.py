import h5py
import numpy as np
import matplotlib.pyplot as plt


def summarize_dataset(file_path):
    """
    Loads the dataset from an HDF5 file, calculates summary statistics,
    and generates plots to visualize the data distributions.
    """
    try:
        with h5py.File(file_path, 'r') as hf:
            # Load data into memory
            angle = hf['angle'][:]
            camera_images = hf['camera_image']

            print("--- Dataset Summary ---")
            print(f"Camera image shape: {camera_images.shape}")

            print("\n--- Angle ---")
            print(f"  Min: {np.min(angle):.4f}")
            print(f"  Max: {np.max(angle):.4f}")
            print(f"  Mean: {np.mean(angle):.4f}")
            print(f"  Std Dev: {np.std(angle):.4f}")

            # --- Visualizations ---
            print("\n--- Generating Plots ---")

            # Plot for Distance and Angle
            plt.figure(figsize=(10, 5))
            plt.suptitle('Data Distributions', fontsize=16)

            plt.hist(angle, bins=50, color='salmon', edgecolor='black')
            plt.title('Angle Distribution')
            plt.xlabel('Angle (radians)')
            plt.ylabel('Frequency')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            dist_plot_path = 'distribution_plots.png'
            plt.savefig(dist_plot_path)
            print(f"Distribution plot saved to: {dist_plot_path}")
            plt.close(fig)

    except FileNotFoundError:
        print(f"Error: The file was not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    # Path to your HDF5 dataset file
    dataset_path = 'images.h5'
    summarize_dataset(dataset_path)
