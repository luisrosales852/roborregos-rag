import os
from dotenv import load_dotenv

load_dotenv()



from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter


script_dir = os.path.dirname(os.path.abspath(__file__))
pdf1_path = os.path.join(script_dir, "knowledge1.pdf")

print("Script directory:", script_dir)
loader = PyPDFLoader(pdf1_path)
docs = loader.load()

all_page_content = [doc.page_content for doc in docs]

embeddings_openai = OpenAIEmbeddings(model="text-embedding-3-large")

# Split first document
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
splits = text_splitter.create_documents(all_page_content)
print(f"Document1 split into {len(splits)} chunks.")

#Vector Store. For now just put them into one vector store, then work on indexing if needed.
print("Trying to use chroma vector store")
print(f"{splits[0]}")
vectorstore1 = Chroma(collection_name="knowledge1_collection",embedding_function=embeddings_openai, persist_directory="/chroma_knowledge1_db")
vectorstore1.add_documents(splits)

              
print("Embedded vectors.")
print("Total Vectors:", vectorstore1._collection.count())
retriever = vectorstore1.as_retriever()

prompt = hub.pull("rlm/rag-prompt")

# LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Question
answer = rag_chain.invoke("What is Task Decomposition?")
print(answer)


