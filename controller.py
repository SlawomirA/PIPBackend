from sqlalchemy.orm import Session

import knn
from database import *
from sqlmodel import Field, Session, SQLModel, select
from fastapi import Depends, HTTPException, status, APIRouter
from models import Data
from utils import create_data
from knn import KNN
import json

router = APIRouter()


def get_knn_instance(db: Session = Depends(get_db)) -> KNN:
    """
    Get KNN instance
    :param db: Database session
    :return: KNN instance
    """
    data = db.exec(select([Data])).all()
    knn_instance = KNN(data, k=5)
    knn_instance.train_model(data)
    return knn_instance


@router.post("/data")
async def create_point(data: Data, db: Session = Depends(get_db)):
    """
    Creates a new data point in the database
    :param data: Data to create
    :param db: Database session
    :return: created data point
    """
    if data.continuous_feature_1 < 0:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Continuous feature 1 must be greater than 0")

    if data.continuous_feature_2 < 0:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Continuous feature 2 must be greater than 0")
    if data.category <= 0:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Category must be greater than 0")

    if (not isinstance(data.continuous_feature_1, float)
            or not isinstance(data.continuous_feature_2, float)):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Continuous features must be floats")

    if not isinstance(data.category, int):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Category must be integer")

    db.add(data)
    db.commit()
    db.refresh(data)
    return data


@router.get("/data")
async def get_points():
    """
    Returns all data points in the database
    :return: all data points in the database
    """
    with Session(engine) as session:
        return session.exec(select([Data])).all()


@router.delete("/data/{record_id}")
async def delete_data(record_id: int):
    """
    Deletes a record from database if it exists
    :param record_id: id of the record to delete
    :return: deleted record
    """
    with Session(engine) as session:
        # Check if the record with the given record_id exists
        data_point = session.exec(select(Data).where(Data.id == record_id)).first()
        if data_point:
            session.delete(data_point)
            session.commit()
            return {"deleted_record_id": record_id}
        else:
            raise HTTPException(status_code=404, detail="Record not found")


@router.get("/predict")
async def predict(x: float, y: float, k: int = 2, model: KNN = Depends(get_knn_instance)):
    """
    Predict the class of a given data point
    :param x: x coordinate of the data point
    :param y: y coordinate of the data point
    :param k: number of nearest neighbors to use to predict the class
    :param model: trained model
    :return: predicted class
    """
    if k < 1:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="K must be greater than 0")

    if not isinstance(x, float) or not isinstance(y, float):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="X and Y must be floats")

    return {"predicted_category": model.predict_category(x, y, k)}


@router.get("/generate")
async def generate_points(x: float = 0, y: float = 0, r: float = 1, amount: int = 2, category: int = 2,
                          db: Session = Depends(get_db)):
    """
    Generates a random data points from a circle of radius r and center point
    equal to x and y
    :param x: center x position
    :param y: center y position
    :param r: radius of the circle
    :param amount: amount of points to generate
    :param category: category of the points
    :param db: database connection
    :return:
    """
    if amount < 0:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Amount must be greater than 0")

    if r < 0:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Radius must be greater than 0")
    if category < 0:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Category must be greater than 0")

    if (not isinstance(x, float)
            or not isinstance(y, float)
            or not isinstance(r, float)):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Features must be floats")

    generated_points = create_data(x, y, r, amount, category)
    [db.add(point) for point in generated_points]
    db.commit()
    [db.refresh(point) for point in generated_points]

    return generated_points


@router.get("/train")
async def train(k: int = 2, db: Session = Depends(get_db)):
    """
    Trains the model
    :param k: Number of training points
    :param db: Database session
    :return: Pair of 2 plots in base64 format
    """
    data = db.exec(select([Data])).all()
    knn_instance = KNN(data, k)
    knn_instance.train_model(data, k)
    return {"k": k, "plot_images": knn_instance.plots()}
