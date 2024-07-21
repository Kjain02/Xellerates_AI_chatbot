import pickle
import os
import numpy as np
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


embeddings_model = OpenAIEmbeddings()

with open('my_list.pkl', 'rb') as file:
    texts = pickle.load(file)

def get_context(query_text, index, k):

    # Query vector to search

    query_vector = embeddings_model.embed_query(query_text)
    query_vector_array = np.array(query_vector).astype("float32")
    query_vector_array = query_vector_array.reshape(1, -1)
    # Perform the similarity search
    # k = 2  # number of nearest neighbors
    distances, indices = index.search(query_vector_array, k)
    # print("Nearest neighbors:", indices)
    # print("Distances:", distances)

    if len(distances[0]) == 0:
        return "No context found"

    context_doc = []
    
    for idx, d in enumerate(distances[0]):
        if d < 0.4:
            context_doc.append(Document(page_content=texts[indices[0][idx]]))

        return context_doc
