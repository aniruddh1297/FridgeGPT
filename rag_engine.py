# rag_engine.py

import os
import json
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

def load_recipes(filepath="data/recipes.jsonl"):
    docs = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            content = (
                f"Title: {data['title']}\n"
                f"Ingredients: {', '.join(data['ingredients'])}\n"
                f"Instructions: {data['instructions']}"
            )
            docs.append(Document(page_content=content, metadata={"title": data["title"]}))
    return docs

def create_vector_store(documents):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(documents, embeddings)
    db.save_local("faiss_index")
    return db

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

def get_meal_suggestions(user_ingredients, mode="normal", stream_handler=None):
    db = load_vector_store()
    retriever = db.as_retriever(search_kwargs={"k": 5})

    mode_style = {
        "normal": "Just suggest meals normally.",
        "student": "Make it cheap and easy to cook.",
        "vegan": "Only suggest vegan meals.",
        "hangover": "Make it greasy and satisfying for a hangover."
    }

    prompt = f"""
You're a creative cooking assistant. The user has these ingredients: {user_ingredients}.
Your job is to suggest two realistic and creative meal ideas using these ingredients.

Use this style: {mode_style.get(mode, 'Just suggest meals normally.')}

Use the context from existing recipes to inspire ideas.
Respond informally and be fun.
"""

    llm = ChatOpenAI(
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        model_name="mistralai/mistral-7b-instruct:free",
        temperature=0.7,
        streaming=True,
        callbacks=[stream_handler] if stream_handler else None
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        chain_type="stuff"
    )

    response = qa_chain.invoke({"query": prompt})
    return response["result"]
