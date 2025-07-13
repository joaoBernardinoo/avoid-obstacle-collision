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
            dist = hf['dist'][:]
            angle = hf['angle'][:]
            lidar_data = hf['lidar_data'][:]
            camera_images = hf['camera_image']

            print("--- Dataset Summary ---")
            print(f"Total samples: {len(dist)}")
            print(f"Camera image shape: {camera_images.shape}")
            print(f"Lidar data shape: {lidar_data.shape}")

            # --- Statistics ---
            print("\n--- Statistical Analysis ---")
            print("\n--- Distance ---")
            print(f"  Min: {np.min(dist):.4f}")
            print(f"  Max: {np.max(dist):.4f}")
            print(f"  Mean: {np.mean(dist):.4f}")
            print(f"  Std Dev: {np.std(dist):.4f}")

            print("\n--- Angle ---")
            print(f"  Min: {np.min(angle):.4f}")
            print(f"  Max: {np.max(angle):.4f}")
            print(f"  Mean: {np.mean(angle):.4f}")
            print(f"  Std Dev: {np.std(angle):.4f}")

            # Handle potential infinity values in lidar data
            lidar_finite = lidar_data[np.isfinite(lidar_data)]
            print("\n--- LiDAR Data (finite values) ---")
            print(f"  Min: {np.min(lidar_finite):.4f}")
            print(f"  Max: {np.max(lidar_finite):.4f}")
            print(f"  Mean: {np.mean(lidar_finite):.4f}")
            print(f"  Std Dev: {np.std(lidar_finite):.4f}")

            # --- Visualizations ---
            print("\n--- Generating Plots ---")

            # Plot for Distance and Angle
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Data Distributions', fontsize=16)

            ax1.hist(dist, bins=50, color='skyblue', edgecolor='black')
            ax1.set_title('Distance Distribution')
            ax1.set_xlabel('Distance')
            ax1.set_ylabel('Frequency')

            ax2.hist(angle, bins=50, color='salmon', edgecolor='black')
            ax2.set_title('Angle Distribution')
            ax2.set_xlabel('Angle (radians)')
            ax2.set_ylabel('Frequency')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            dist_plot_path = 'distribution_plots.png'
            plt.savefig(dist_plot_path)
            print(f"Distribution plot saved to: {dist_plot_path}")
            plt.close(fig)

            # Plot for LiDAR data
            lidar_mean_per_sample = np.mean(lidar_data, axis=1)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(lidar_mean_per_sample, bins=50,
                    color='lightgreen', edgecolor='black')
            ax.set_title('Distribution of Mean LiDAR Values')
            ax.set_xlabel('Mean LiDAR Value per Sample')
            ax.set_ylabel('Frequency')
            lidar_plot_path = 'lidar_distribution.png'
            plt.savefig(lidar_plot_path)
            print(f"LiDAR distribution plot saved to: {lidar_plot_path}")
            plt.close(fig)

    except FileNotFoundError:
        print(f"Error: The file was not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    # Path to your HDF5 dataset file
    dataset_path = './controllers/cnn/cnn_dataset/cnn_dataset.h5'
    summarize_dataset(dataset_path)
