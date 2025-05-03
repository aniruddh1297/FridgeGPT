# build_index.py
from rag_engine import load_recipes, create_vector_store

docs = load_recipes("data/recipes.jsonl")
create_vector_store(docs)
print("✅ FAISS vector store created and saved.")
