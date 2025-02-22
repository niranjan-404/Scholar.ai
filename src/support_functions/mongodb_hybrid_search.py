from dotenv import load_dotenv
from load_models.model_configurations import load_embedding_model
from langchain.schema.document import Document

load_dotenv()

embedding_model = load_embedding_model()

def weighted_reciprocal_rank(doc_lists, weights=None, c=60):

    if weights is None:
        weights = [1] * len(doc_lists)
    
    if len(doc_lists) != len(weights):
        raise ValueError("Number of rank lists must be equal to the number of weights.")

    doc_source_map = {}
    for doc_list in doc_lists:
        for doc in doc_list:
            doc_source_map[doc["text"]] = {"file":doc.get('source', ''),"page_number": doc.get('page', 0)+1}

    extracted_documents = set()
    rrf_score_dic = {}
    
    for doc_list, weight in zip(doc_lists, weights):
        for rank, doc in enumerate(doc_list, start=1):
            doc_text = doc["text"]
            
            extracted_documents.add(doc_text)
            
            rrf_score = weight * (1 / (rank + c))
            rrf_score_dic[doc_text] = rrf_score_dic.get(doc_text, 0) + rrf_score

    sorted_documents = sorted(extracted_documents, key=lambda x: rrf_score_dic[x], reverse=True)

    sorted_docs = [
        {"text": page_content, "source": doc_source_map.get(page_content, "")}
        for page_content in sorted_documents
    ]


    return sorted_docs

def mongodb_hybrid_search(query, top_k, collection):
    """
    Perform a hybrid search on a MongoDB collection using vector and keyword search.
    
    Returns:
        List of Document objects from combined search results.
    """
    num_documents = collection.count_documents({"embedding": {"$exists": True}})
    if num_documents < top_k:
        top_k = num_documents

    #vector search
    query_vector = embedding_model.embed_query(query)
    numCandidates = collection.count_documents({})
    vector_results = collection.aggregate([
        {
            "$vectorSearch": {
                "queryVector": query_vector,
                "path": "embedding",
                "numCandidates": numCandidates,
                "limit": top_k,
                "index": "vector_search_index"
            },
        },
        {
            "$project": {
                "_id": 1,
                "page": 1,
                "text": 1,
                "source": 1,  
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ])
    x = list(vector_results)

    keyword_results = collection.aggregate([
        {
            "$search": {
                "index": "keyword_search_index", 
                "text": { 
                    "query": query, 
                    "path": "text" 
                } 
            }
        },
        {"$addFields": {
            "score": {"$meta": "searchScore"},
            "source": "$source" ,
            "page": "$page"
        }},
        {"$limit": top_k}
    ])

    y = list(keyword_results)

    doc_lists = [x, y]
    for i in range(len(doc_lists)):
        doc_lists[i] = [
            {"text": doc["text"], "source": doc.get("source", ""), "score": doc["score"]}
            for doc in doc_lists[i]
        ]
   
    combined_documents = weighted_reciprocal_rank(doc_lists)
    
    documents = [Document(page_content=doc["text"], metadata={"source": doc["source"]}) for doc in combined_documents]
    
    return documents