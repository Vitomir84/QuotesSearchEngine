import streamlit as st
from quotes_search_engine import QuoteSearchEngine
from data_reader import load_quotes_from_csv

# Cache the prompts data to avoid reloading every time
@st.cache_data
def load_quotes():
    # quotes_path = "data/quotes.csv"
    quotes_path = "hf://datasets/jstet/quotes-500k/quotes.csv" # path on hugging face
    return load_quotes_from_csv(quotes_path)

# Cache the search engine initialization
@st.cache_resource
def get_search_engine():
    search_engine = QuoteSearchEngine()
    quotes = load_quotes()
    search_engine.add_quotes_to_vector_database(quotes)
    return search_engine

# Initialize search engine only once
search_engine = get_search_engine()

# Streamlit App Interface
st.title("Quote Search Engine")
st.write("Search for similar quotes using the local search engine.")

# Input for the user's prompt
query_input = st.text_input("Enter your quote or phrase:")

# Number of similar prompts to retrieve (k)
k = st.number_input("Number of similar quotes to retrieve:", min_value=1, max_value=10, value=3)

# Button to trigger search
if st.button("Search Quotes"):
    if query_input:
        print(f'Search engine is searching the most similar quotes for query {query_input}')
        similar_quotes, distances = search_engine.most_similar(query_input, top_k=k)
        print(f'Those are: {similar_quotes}, {distances}')

        # Format and display search results
        st.write(f"Search Results: ")
        for i, (prompt, distance) in enumerate(zip(similar_quotes, distances)):
            st.write(f"{i+1}. Prompt: {prompt}, Distance: {distance}")
            print(f'Those are: {prompt}, {distance}')
    else:
        st.error("Please enter a quote or phrase.")

# Additional functionality for vector similarity
st.write("---")
st.write("### Vector Similarities")

if st.button("Retrieve All Vector Similarities"):
    if query_input:
        query_embedding = search_engine.model.encode([query_input])  # Encode the prompt to a vector
        all_similarities = search_engine.cosine_similarity(query_embedding, search_engine.index)
        st.write(f"Vector Similarities: {all_similarities}")
    else:

        st.error("Please enter a quote or phrase.")
