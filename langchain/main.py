import os
from dotenv import load_dotenv

load_dotenv()



import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector


#### INDEXING ####

# Load Documents Prueba
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

# Embed
embeddings = OpenAIEmbeddings()
connection_string = os.getenv("POSGRESS_URL")
collection_name = "agents_blog_rag"

vectorstore = PGVector.from_documents(
    documents=splits,
    embedding=embeddings,
    collection_name=collection_name,
    connection_string=connection_string,
)

retriever = vectorstore.as_retriever()

#### RETRIEVAL and GENERATION ####

# Prompt
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


#Continuing so rag2

# Documents
question = "What kinds of pets do I like?"
document = "My favorite pet is a cat."

