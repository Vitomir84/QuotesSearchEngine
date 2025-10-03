---
title: Search Engine
emoji: 🔥
colorFrom: green
colorTo: red
sdk: streamlit
sdk_version: 1.39.0
app_file: app.py
pinned: false
short_description: Semantic Search engine with Faiss
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-

### For local deployment run 
```
fast_api.py
```
Script creates swagger app with endpoints on [localhost:8084](http://127.0.0.1:8084/docs). First endpoint return the top k semanticaly most similar prompts with query quotes. Second endpoint returns all similarites with query (only applicable for very small datasets).

### Data Ingestion

```
data_reader.py
```
creates data of various quotes for encoding into vector database, from prompt-picture dataset. 
Local database encoded only 11000 prompts.
Faiss index that is used is small and not optimized, used for experimental datasets. Search is brute force, not optimised. 

### Streamlit
```
streamlit run app.py
```
Should be run for streamlit app, it can be assessed locally on http://localhost:8501.

### Docker
```
docker build -t my-streamlit-app .
```
from main dir

