import warnings
from context import get_context
from chatbot import chain
import faiss


warnings.filterwarnings("ignore", category=UserWarning)


# Load the FAISS index
index = faiss.read_index("embeddings.index")

# Start the chat from here
while True:
    user_input = input("You: ")
    if user_input.lower() in ["thank you", "bye"]:
        break
    context_doc = get_context(user_input, index, 2)
    answer = chain.run({"input_documents": context_doc, "question": user_input})
    print(answer)