import os
from dotenv import load_dotenv
import pymongo
from pymongo.server_api import ServerApi

load_dotenv()
db_client = os.getenv("MONGODB_CONNECTION_STRING")

mongo_client = pymongo.MongoClient(db_client, server_api=ServerApi("1"))