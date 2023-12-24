from dataclasses import Field
from typing import Optional

from sqlmodel import Field, Session, SQLModel, create_engine, select


class Data(SQLModel, table=True):
    """
    Data model for stored data point
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    continuous_feature_1: float
    continuous_feature_2: float
    category: int
