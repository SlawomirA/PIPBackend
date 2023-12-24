from fastapi import FastAPI
from database import *
from controller import *
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    """
    Called when the server starts
    :return:
    """
    create_db_and_tables()


app.include_router(router, prefix="/api", tags=["api"])
