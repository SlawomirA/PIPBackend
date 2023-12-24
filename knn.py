import random
from typing import List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models import *


class KNN:
    def __init__(self, data: List[Data], k: int = 2):
        """
        Initialize the KNN
        :param data: List of Data points
        :param k: Number of neighbors
        """
        self.data = None
        self.x = None
        self.y = None
        self.k = None

        self.scaler = StandardScaler()
        self.x_standardized = None
        self.knn_classifier = None
        self.current_model = None

        self.load_data(data, k)
        self.train_model(data, k)

    def load_data(self, data: List[Data], k: int = 2) -> None:
        """
        Load the data
        :param data: List of Data points
        :param k: Number of neighbors
        :return:
        """
        # Load data and create numpy arrays
        self.data = data
        self.x = np.array([[point.continuous_feature_1, point.continuous_feature_2] for point in data])
        self.y = np.array([point.category for point in data])
        self.k = k

        self.x_standardized = self.scaler.fit_transform(self.x)

    def train_model(self, data: List[Data], k: int = 2):
        """
        Trains the model using the provided data and fits the KNN classifier
        :param data: List of Data to be used for training
        :param k: Number of neighbors to use for fitting the KNN classifier
        :return:
        """
        # Load data and create numpy arrays
        self.load_data(data, k)

        # Create and train the k-nearest neighbors classifier
        self.knn_classifier = KNeighborsClassifier(n_neighbors=k)
        self.knn_classifier.fit(self.x_standardized, self.y)

        # Save the trained model to the attribute
        self.current_model = self.knn_classifier

    def predict_category(self, x: float, y: float, k: Optional[int] = None) -> int:
        """
        Predicts the category of a data point using KNN classifier
        :param x: x coordinate of the data point
        :param y: y coordinate of the data point
        :param k: Number of neighbors to use for fitting the KNN classifier
        :return: Predicted category of the data point
        """
        # Standardize the new point
        new_point_standardized = self.scaler.transform([[x, y]])

        # Use the provided k parameter or default to the trained k
        k_to_use = k if k is not None else self.k

        # Create a new KNN classifier with the specified k
        knn_classifier = KNeighborsClassifier(n_neighbors=k_to_use)
        knn_classifier.fit(self.x_standardized, self.y)

        # Predict the category using the new KNN classifier
        predicted_category = knn_classifier.predict(new_point_standardized)

        return int(predicted_category[0])

    # def save_last_trained_model(self, filename: str) -> None:
    #     """
    #     Saves the trained model to the given filename
    #     :param filename: Filename to save the trained model
    #     :return:
    #     """
    #     if self.current_model is not None:
    #         joblib.dump(self.current_model, filename)
    #     else:
    #         print("No model has been trained yet.")
    #
    # def load_last_trained_model(self, filename: str) -> None:
    #     """
    #     Loads the trained model from the given filename
    #     :param filename: Filename to load the trained model
    #     :return:
    #     """
    #     loaded_model = joblib.load(filename)
    #     if isinstance(loaded_model, KNeighborsClassifier):
    #         self.current_model = loaded_model
    #     else:
    #         print("Invalid model file.")

    def plots(self, xlabel: Optional[str] = "Continuous Feature 1",
              ylabel: Optional[str] = "Continuous Feature 2") -> list[str]:
        """
        Plot the plots for a specific data
        :param xlabel: label for the x-axis
        :param ylabel: label for the y-axis
        :return:
        """
        _, axs = plt.subplots(ncols=2, figsize=(12, 5))
        plot_images = []

        # Create 2 plots for uniform and distance weights
        for ax, weights in zip(axs, ("uniform", "distance")):
            self.current_model.set_params(weights=weights)
            disp = DecisionBoundaryDisplay.from_estimator(
                self.current_model,
                self.x_standardized,
                response_method="predict",
                plot_method="pcolormesh",
                xlabel=xlabel,
                ylabel=ylabel,
                shading="auto",
                alpha=0.5,
                ax=ax,
            )

            # Create scatterplot for all the points
            scatter = disp.ax_.scatter(self.x_standardized[:, 0], self.x_standardized[:, 1], c=self.y, edgecolors="k")
            disp.ax_.legend(
                scatter.legend_elements()[0],
                np.unique(self.y),
                loc="lower left",
                title="Classes",
            )

            # Add ID labels around the circles
            for data_point, x, y, point_id in zip(self.data, self.x_standardized[:, 0], self.x_standardized[:, 1],
                                                  range(0, len(self.x_standardized))):
                disp.ax_.text(x - 0.1, y, str(point_id), fontsize=8, ha='center', va='center', color='orangered')

            _ = disp.ax_.set_title(f"{len(np.unique(self.y))}-Class classification\n(k={self.k}, weights={weights!r})")

        # plt.show()
        plot_image_stream = io.BytesIO()
        plt.savefig(plot_image_stream, format='png')
        plot_image_stream.seek(0)
        plot_image_base64 = base64.b64encode(plot_image_stream.read()).decode('utf-8')
        plt.close()  # Close the current plot to avoid overlapping with the next one

        plot_images.append(plot_image_base64)

        return plot_images

# if __name__ == "__main__":
#     # Create some sample data points
#     sample_data = [
#         Data(continuous_feature_1=1.0, continuous_feature_2=2.0, category=0),
#         Data(continuous_feature_1=2.0, continuous_feature_2=3.0, category=1),
#         Data(continuous_feature_1=3.0, continuous_feature_2=4.0, category=0),
#         Data(continuous_feature_1=5.0, continuous_feature_2=7.0, category=1)
#     ]
#
#     # Create an instance of the KNN class with the sample data
#     current_model = KNN(data=sample_data)
#
#     # Visualize the decision boundaries
#     current_model.plots(xlabel="Feature 1", ylabel="Feature 2")
