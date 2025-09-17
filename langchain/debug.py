import os
from dotenv import load_dotenv

load_dotenv()

import bs4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate

# Load Documents
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

# Split into larger chunks to get fewer total chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)  # Doubled chunk size
splits = text_splitter.split_documents(docs)

# Use only first 5 chunks for testing
splits = splits[:5]
print(f"Using {len(splits)} chunks for faster processing")

# Embed
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

# Simple prompt (skip hub.pull)
prompt = PromptTemplate(
    template="""Answer the question based on this context:

Context: {context}

Question: {question}

Answer:""",
    input_variables=["context", "question"]
)

# LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Question
print("Processing question...")
answer = rag_chain.invoke("What is Task Decomposition?")
print("\nAnswer:")
print(answer)