import os
from dotenv import load_dotenv
import traceback

try:
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    print(f"API Key loaded: {'Yes' if openai_api_key else 'No'}")
    
    import bs4
    from langchain import hub
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_community.vectorstores import Chroma
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    print("All imports successful")

    #### INDEXING ####
    print("Starting document loading...")
    
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
    print(f"Documents loaded: {len(docs)} documents")

    # Split
    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"Document splits created: {len(splits)} chunks")

    # Embed
    print("Creating embeddings...")
    vectorstore = Chroma.from_documents(documents=splits, 
                                        embedding=OpenAIEmbeddings(api_key=openai_api_key))
    print("Vector store created successfully")

    retriever = vectorstore.as_retriever()
    print("Retriever created")

    #### RETRIEVAL and GENERATION ####
    print("Setting up RAG chain...")
    
    # Prompt
    prompt = hub.pull("rlm/rag-prompt")
    print("Prompt loaded")

    # LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key=openai_api_key)
    print("LLM initialized")

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
    print("RAG chain created")

    # Question
    print("Invoking RAG chain...")
    answer = rag_chain.invoke("What is Task Decomposition?")
    print("Answer received:")
    print(answer)

except Exception as e:
    print(f"Error occurred: {str(e)}")
    print("Full traceback:")
    traceback.print_exc()