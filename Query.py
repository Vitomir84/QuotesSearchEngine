from pydantic import BaseModel
from typing import List



class Query(BaseModel):
    
    id: str
    quote: str

class Query_Multiple(BaseModel):
    prompt: List[Query]


class SimilarQuote(BaseModel):
    prompt: str
    distance: float  

class SearchResponse(BaseModel):
    results: List[SimilarQuote]

class QuoteVector(BaseModel):
    vector: int
    distance: float

class VectorResponse(BaseModel):
    results: List[QuoteVector] 