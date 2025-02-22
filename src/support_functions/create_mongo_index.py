from pymongo.operations import SearchIndexModel
import os
from dotenv import load_dotenv


load_dotenv("env.dev")

def create_keyword_search_index(collection,field ="text"):

    search_model = SearchIndexModel(
            definition=   {
                "mappings": {
                    "fields": {
                        field: {
                            "type": "string"
                        }
                    }
                }
            },
            name="keyword_search_index"
        )
    
    collection.create_search_index(model=search_model)

    Status = "PENDING"                       
    while Status != "READY":
            indexes = collection.list_search_indexes()
            for i in indexes:
                    if i['name'] == 'keyword_search_index':
                        Status = i["status"]   

def create_vector_index(collection):

    collection.create_search_index(
    {"definition":
        {"mappings": {"dynamic": False, "fields": {
            "embedding" : {
                "dimensions": 3072,
                "similarity": "cosine",
                "type": "knnVector"
                }}}},
     "name": "vector_search_index"
    }
    )
    Status = "PENDING"                       
    while Status != "READY":
            indexes = collection.list_search_indexes()
            for i in indexes:
                    if i["name"] == "vector_search_index":
                            Status = i["status"]   

