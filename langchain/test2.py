import os
from dotenv import load_dotenv

load_dotenv()

import bs4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate

print("Starting RAG setup with persistent storage...")

# Set up embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Check if we already have a database
persist_dir = "./chroma_langchain_db"
if os.path.exists(persist_dir):
    print("Loading existing vector store...")
    vector_store = Chroma(
        collection_name="rag_collection",
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    print("✅ Loaded existing vector store")
else:
    print("Creating new vector store...")
    
    # Load and process documents
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()
    
    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # Create persistent vector store
    vector_store = Chroma(
        collection_name="rag_collection", 
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    
    # Add documents in batches
    batch_size = 10
    for i in range(0, len(splits), batch_size):
        batch = splits[i:i + batch_size]
        vector_store.add_documents(documents=batch)
        print(f"   Added batch {i//batch_size + 1}/{(len(splits) + batch_size - 1)//batch_size}")
    
    print("✅ Vector store created and saved")

# Set up retriever and chain
retriever = vector_store.as_retriever()

prompt = PromptTemplate(
    template="""Answer the question based on the context:

Context: {context}

Question: {question}

Answer:""",
    input_variables=["context", "question"]
)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Process question
answer = rag_chain.invoke("What is Task Decomposition?")

print("\n" + "="*60)
print("ANSWER:")
print("="*60)
print(answer)
print("="*60)