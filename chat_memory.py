from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="history",
    return_messages=False
)