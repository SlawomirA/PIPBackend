from typing import List, Tuple
import random
from models import Data


def generate_data(x: float, y: float, r: float, amount: int) -> list[tuple[float, float]]:
    """
    Generate the points for a specific category
    :param x: x coordinate of the circle
    :param y: y coordinate of the circle
    :param r: radius of the circle
    :param amount: amount of the points to generate
    :return: generated points
    """
    points: List[Tuple[float, float]] = []
    while len(points) < amount:
        point: Tuple[float, float] = (random.uniform(x - r, x + r), random.uniform(y - r, y + r))
        if point not in points:
            points.append(point)

    return points


def create_data(x: float, y: float, r: float, amount: int, category: int) -> List[Data]:
    """
    Generate the points for a specific category
    :param self:
    :param x: x coordinate of the circle
    :param y: y coordinate of the circle
    :param r: radius for the circle
    :param amount: amount of the points
    :param category: category of the points to generate
    :return: generated points
    """
    return [Data(continuous_feature_1=a, continuous_feature_2=b, category=category)
            for a, b in generate_data(x, y, r, amount)]