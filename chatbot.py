from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI




memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="question",
    return_messages=True,
)


template_str = """
### Instruction: You're a customer support agent that is talking to a customer who wanted to know about Xellerates AI.
Your job is to help customer with their queries related to general information about Xellerates AI, sign-in, sign-up, startups.
Use only the following context and chat history to answer questions.
Be as detailed as possible, but don't make up any information
that's not from the context. If not context is found, say that you don't know and it is not related to Xellerates AI.
{context}
If you don't know the answer - say that you don't know. Keep your replies short, compassionate and informative.

{chat_history}

"""

system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context", "chat_history"],
        template=template_str,
    )
)

human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["question"],
        template="{question}",
    )
)
messages = [system_prompt, human_prompt]

prompt_template = ChatPromptTemplate(
    input_variables=["context", "chat_history", "question"],
    messages=messages,
)

chat_model = ChatOpenAI(model="gpt-4o", temperature=0)



chain = load_qa_chain(
    chat_model, chain_type="stuff", prompt=prompt_template, memory=memory, verbose=False
)





