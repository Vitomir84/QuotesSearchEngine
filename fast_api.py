import sys
import os
import copy
import uvicorn
import socket
import logging
import datetime
from models.quotes_search_engine import QuoteSearchEngine
from models.data_reader import load_quotes_from_csv
from models.Query import Query, Query_Multiple, SearchResponse, SimilarQuote, QuoteVector, VectorResponse
from decouple import config
from fastapi import FastAPI, HTTPException, Depends, Body
from sentence_transformers import SentenceTransformer



# quotes_path = r"C:\Users\jov2bg\Desktop\QuotesSearch\search_engine\data\quotes.csv"
quotes_path = "hf://datasets/jstet/quotes-500k/quotes.csv" # path on hugging face


app = FastAPI(title="Search Prompt Engine", description="API for prompt search", version="1.0")

quotes = load_quotes_from_csv(quotes_path)
search_engine = QuoteSearchEngine()
search_engine.add_quotes_to_vector_database(quotes[:10000])

@app.get("/")
def read_root():
    return {"message": "Quote Search Engine is running!"}

@app.post("/search/")
async def search_prompts(query: Query, k: int = 3):
    print(f'Prompt: {query}')
    similar_quotes, distances = search_engine.most_similar(query.quote, top_k=k)
    print(f'Similar Quotes {similar_quotes}')
    print(f'Distances {distances}')
    print(40*'****')
    # Format the response
    response = [
        SimilarQuote(prompt=prompt, distance=float(distance)) 
        for prompt, distance in zip(similar_quotes, distances)
    ]
    
    return SearchResponse(results=response)

@app.post("/all_vectors_similarities/")
async def all_vectors(query: Query):

    query_embedding = search_engine.model.encode([query.quote])  # Encode the prompt to a vector
    all_similarities = search_engine.cosine_similarity(query_embedding, search_engine.index)
    print(f'Prompt: {query}')
    print(f'All Vector Similarities: {all_similarities}')
    print(40*'****')
    response = [
        QuoteVector(vector=index, distance=float(distance)) 
        for index, distance in enumerate(all_similarities)
    ]
    return VectorResponse(results=response)

if __name__ == "__main__":
    # Server Config
    # SERVER_HOST_IP = socket.gethostbyname(socket.gethostname())
    SERVER_HOST_IP = socket.gethostbyname("localhost") # for local deployment
    SERVER_PORT = int(8084)
    uvicorn.run(app, host=SERVER_HOST_IP, port=SERVER_PORT)