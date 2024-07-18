from langchain.chains import RetrievalQA  
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize a LangChain object for chatting with the LLM
# without knowledge from Pinecone.
llm = ChatOpenAI(
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    model_name="gpt-3.5-turbo",
    temperature=0.0
)

# Initialize a LangChain object for retrieving information from Pinecone.
knowledge = PineconeVectorStore.from_existing_index(
    index_name="docs-rag-chatbot",
    namespace="wondervector5000",
    embedding=OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
)

# Initialize a LangChain object for chatting with the LLM
# with knowledge from Pinecone. 
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=knowledge.as_retriever()
)

# Define a few questions about the WonderVector5000.
query1 = """What are the first 3 steps for getting started 
with the WonderVector5000?"""

query2 = """The Neural Fandango Synchronizer is giving me a 
headache. What do I do?"""

# Send each query to the LLM twice, first with relevant knowledge from Pincone 
# and then without any additional knowledge.
print("Query 1\n")
print("Chat with knowledge:")
print(qa.invoke(query1).get("result"))
print("\nChat without knowledge:")
print(llm.invoke(query1).content)
print("\nQuery 2\n")
print("Chat with knowledge:")
print(qa.invoke(query2).get("result"))
print("\nChat without knowledge:")
print(llm.invoke(query2).content)

