from typing import Optional

from sqlmodel import Field, Session, SQLModel, create_engine, select

sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, echo=True, connect_args=connect_args)


def create_db_and_tables():
    """
    Creates the database and tables
    :return:
    """
    SQLModel.metadata.create_all(engine)


def get_db():
    """
    Gets database connection
    :return:
    """
    db = Session(engine)
    try:
        yield db
    finally:
        db.close()
