import dataset_utils as dataset_utils
import torch
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


class SpatialGrid:
    def __init__(
        self,
        dataset_folder: str,
        cell_size: float,
    ) -> None:
        """
        Initialize the SpatialGrid object.

        Args:
            dataset_folder (str): The path for training dataset.
            cell_size (float): The size of each square cell.
        """
        self.dataset_folder: str = dataset_folder
        self.cell_size: float = cell_size
        self.data: pd.DataFrame = self.load_data()
        self.x: str = "UTM_EAST"
        self.y: str = "UTM_NORTH"
        self.grid: dict = self.create_grid()
        self.grid_iter = iter(self.grid.values())

    def load_data(self) -> pd.DataFrame:
        """
        Get the data information from data path and save them as a dataframe.

        Returns:
            dataframe: A pandas DataFrame that contains all the location and direction information for the images in training set.
        """
        print(f"==> Searching training images in {self.dataset_folder}")

        images_paths = dataset_utils.read_images_paths(self.dataset_folder)
        print(f"==> Found {len(images_paths)} images")

        print(
            "==> For each image, get its UTM east, UTM north and heading from its path"
        )
        images_metadatas = [p.split("@") for p in images_paths]
        # field 1 is UTM east, field 2 is UTM north, field 9 is heading
        utmeast_utmnorth_heading = [[m[1], m[2], m[9]] for m in images_metadatas]
        utmeast_utmnorth_heading = np.array(utmeast_utmnorth_heading).astype(np.float64)
        dataframe = pd.DataFrame(
            utmeast_utmnorth_heading, columns=["UTM_EAST", "UTM_NORTH", "HEADING"]
        )
        dataframe["HEADING"] = (
            dataframe["HEADING"] - 180
        )  # set the range from -180 to 180
        return dataframe

    def create_grid(self) -> dict:
        """
        Create a grid of square cells based on the given cell size.

        Returns:
            dict: A dictionary where keys are cell coordinates (x, y) and values are DataFrames containing data in each cell.
        """
        min_x, max_x = self.data[self.x].min(), self.data[self.x].max()
        min_y, max_y = self.data[self.y].min(), self.data[self.y].max()

        # Calculate the number of cells in both x and y directions
        num_cells_x = int((max_x - min_x) / self.cell_size) + 1
        num_cells_y = int((max_y - min_y) / self.cell_size) + 1

        grid = {}

        print("==> Loading the grids")
        for i in tqdm(range(num_cells_x), desc="generating grid x", leave=True):
            for j in tqdm(range(num_cells_y), desc="generating grid y", leave=False):
                cell_min_x = min_x + i * self.cell_size
                cell_max_x = cell_min_x + self.cell_size
                cell_min_y = min_y + j * self.cell_size
                cell_max_y = cell_min_y + self.cell_size

                # Extract the subset of data within the current cell
                cell_data = self.data[
                    (self.data[self.x] >= cell_min_x)
                    & (self.data[self.x] < cell_max_x)
                    & (self.data[self.y] >= cell_min_y)
                    & (self.data[self.y] < cell_max_y)
                ]

                # Only store the non-empty cell data in the grid dictionary
                if not cell_data.empty:
                    cell_data.reset_index(drop=True, inplace=True)
                    grid[(i, j)] = cell_data

        return grid

    def get_cell_data(self, cell_x, cell_y) -> pd.DataFrame:
        """
        Retrieve the subset of data within a specific cell.

        Args:
            cell_x (int): The x-coordinate of the cell.
            cell_y (int): The y-coordinate of the cell.

        Returns:
            pd.DataFrame: The subset of data within the specified cell.
        """
        return self.grid.get((cell_x, cell_y), pd.DataFrame())

    def get_max_cell_coordinates(self):
        """
        Retrieve the maximum range of cell_x and cell_y in the grid.

        Returns:
            tuple: A tuple containing the maximum cell_x and cell_y.
        """
        max_x = max(coord[0] for coord in self.grid.keys())
        max_y = max(coord[1] for coord in self.grid.keys())
        return max_x, max_y

    def show_cell_info(self):
        """
        Display the available cell coords and print the number of cells
        """
        print(f"The number of cells are: {len(self.grid.keys())}")
        print(f"They are: {self.grid.keys()}")

    def count_samples_in_cells(self):
        """
        Count the number of samples in each cell.

        Returns:
            dict: A dictionary where keys are cell coordinates (x, y) and values are the number of samples in each cell.
        """
        cell_counts = {}
        for cell_coords, cell_data in self.grid.items():
            cell_counts[cell_coords] = len(cell_data)
        return cell_counts

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.grid_iter)


class HeatmapAnalyzer:
    def __init__(
        self,
        std_dev: float = 40,
        direction_range: float = np.pi / 4,
        grid_resolution: float = 0.1,
    ):
        """
        Initialize the Heatmap Analyzer

        Params:
            std_dev: The range in length (gaussian kernel) that a point can see.
            direction_range: The range in angle that a point can see.
            grid_resolution: Grid resolution of the heatmap.
        """
        self.std_dev = std_dev
        self.direction_range = direction_range
        self.grid_resolution = grid_resolution
        self.probability_density = None

    def generate_probability_density_heatmap(self, dataframe) -> None:
        """
        Generate a probability density heatmap for a DataFrame of samples with direction and coordinates.

        Args:
            dataframe (pd.DataFrame): Input DataFrame with columns 'UTM_EAST', 'UTM_NORTH', 'HEADING'.

        Returns:
            None (displays the heatmap).
        """
        # Create a grid that covers the cell area
        self.x_min = dataframe["UTM_EAST"].min()
        self.x_max = dataframe["UTM_EAST"].max()
        self.y_min = dataframe["UTM_NORTH"].min()
        self.y_max = dataframe["UTM_NORTH"].max()
        x_grid = np.arange(self.x_min, self.x_max, self.grid_resolution)
        y_grid = np.arange(self.y_min, self.y_max, self.grid_resolution)
        xx, yy = np.meshgrid(x_grid, y_grid)

        # Initialize an empty grid for probability density
        self.probability_density = np.zeros_like(xx)

        # Calculate probability density for each sample
        for _, row in dataframe.iterrows():
            x_sample = row["UTM_EAST"]
            y_sample = row["UTM_NORTH"]
            ca_sample = row["HEADING"]

            # Calculate the direction distance for each grid point
            direction_distance = np.arctan2(xx - x_sample, yy - y_sample)

            min_theta = math.radians(ca_sample) - self.direction_range
            max_theta = math.radians(ca_sample) + self.direction_range

            valid_direction = (
                ((direction_distance >= min_theta) & (direction_distance <= max_theta))
                | (direction_distance >= min_theta + 2 * np.pi)
                | (direction_distance <= max_theta - 2 * np.pi)
            )

            # Calculate Gaussian kernel values
            kernel_values = (
                np.exp(-((xx - x_sample) ** 2) / (2 * self.std_dev**2))
                * np.exp(-((yy - y_sample) ** 2) / (2 * self.std_dev**2))
                * valid_direction
            )

            # Add the kernel values to the probability density grid
            self.probability_density += kernel_values

    def plot_heatmap(self) -> None:
        """
        Create a heatmap of the probability density
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(
            self.probability_density,
            extent=(self.x_min, self.x_max, self.y_min, self.y_max),
            origin="lower",
            cmap="viridis",
        )
        plt.colorbar(label="Probability Density")
        plt.xlabel("Easting")
        plt.ylabel("Northing")
        plt.title("Probability Density Heatmap")
        plt.show()

    def find_top_n_areas_centers(self, n: int, threshold: float = 50) -> list:
        """
        Find the centers of the top N highest areas in a divided grid of the heatmap.

        Args:
            n (int): Number of divisions along each axis.
            threshold (float): Minimum value to consider a point.

        Returns:
            list of tuples: List of (x, y) coordinates of the centers of the top N areas.
        """
        if self.probability_density is None:
            raise ValueError(
                "Heatmap not generated. Call generate_probability_density_heatmap() first."
            )

        cell_size_x = (self.x_max - self.x_min) / n
        cell_size_y = (self.y_max - self.y_min) / n

        centers = []

        for i in range(n):
            for j in range(n):
                # Define the boundaries of the current cell
                x_start = self.x_min + j * cell_size_x
                x_end = self.x_min + (j + 1) * cell_size_x
                y_start = self.y_min + i * cell_size_y
                y_end = self.y_min + (i + 1) * cell_size_y

                # Extract the sub-heatmap for the current cell
                sub_heatmap = self.probability_density[
                    int((y_start - self.y_min) / self.grid_resolution) : int(
                        (y_end - self.y_min) / self.grid_resolution
                    ),
                    int((x_start - self.x_min) / self.grid_resolution) : int(
                        (x_end - self.x_min) / self.grid_resolution
                    ),
                ]

                # Find the index of the highest value point in the sub-heatmap
                max_index = np.unravel_index(np.argmax(sub_heatmap), sub_heatmap.shape)

                # Calculate the coordinates of the highest value point within the cell
                max_x = x_start + max_index[1] * self.grid_resolution
                max_y = y_start + max_index[0] * self.grid_resolution

                # Check if the value at the highest point is greater than the threshold
                if (
                    self.probability_density[
                        int((max_y - self.y_min) / self.grid_resolution),
                        int((max_x - self.x_min) / self.grid_resolution),
                    ]
                    >= threshold
                ):
                    centers.append((max_x, max_y))

        return centers

    def plot_heatmap_with_top_centers(self, n) -> None:
        """
        Plot the heatmap with top centers
        """
        if self.probability_density is None:
            raise ValueError(
                "Heatmap not generated. Call generate_probability_density_heatmap() first."
            )

        top_n_centers = self.find_top_n_areas_centers(n)

        plt.figure(figsize=(10, 8))
        plt.imshow(
            self.probability_density,
            extent=(self.x_min, self.x_max, self.y_min, self.y_max),
            origin="lower",
            cmap="viridis",
        )
        plt.colorbar(label="Probability Density")

        # Plot the centers of the top N areas as red dots
        for center in top_n_centers:
            plt.plot(center[0], center[1], "ro", markersize=10)

        plt.xlabel("Easting")
        plt.ylabel("Northing")
        plt.title("Probability Density Heatmap")

        plt.show()


def main():
    folder = "/media/dszhang/data/vpr_ws/datasets/SF-XL/processed/train"
    grid_data = SpatialGrid(folder, 100)
    for i in grid_data:
        print(i)


if __name__ == "__main__":
    main()
