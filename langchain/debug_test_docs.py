import os
from dotenv import load_dotenv

load_dotenv()

print("1. Environment loaded")

import re
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document

print("2. Imports completed")

def clean_pdf_text(text):
    # Replace multiple whitespace characters (spaces, tabs, newlines) with single space
    cleaned_text = re.sub(r'\s+', ' ', text)
    
    # Remove leading and trailing whitespace
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text

def clean_document_list(docs):
    """
    Clean a list of Document objects, preserving metadata
    """
    cleaned_docs = []
    
    for doc in docs:
        # Clean the page content
        cleaned_content = clean_pdf_text(doc.page_content)
        
        # Create new document with cleaned content and original metadata
        cleaned_doc = Document(
            page_content=cleaned_content,
            metadata=doc.metadata
        )
        cleaned_docs.append(cleaned_doc)
    
    return cleaned_docs

print("3. Functions defined")

script_dir = os.path.dirname(os.path.abspath(__file__))
pdf1_path = os.path.join(script_dir, "knowledge1.pdf")

print("Script directory:", script_dir)
print("PDF path:", pdf1_path)
print("PDF exists:", os.path.exists(pdf1_path))

print("4. About to load PDF")
loader = PyPDFLoader(pdf1_path)
docs = loader.load()
print(f"5. PDF loaded, {len(docs)} pages")

print("6. About to create embeddings")
embeddings_openai = OpenAIEmbeddings(model="text-embedding-3-large")
print("7. Embeddings created")

# Split first document
print("8. About to split documents")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
splits = text_splitter.split_documents(docs)
print(f"9. Documents split into {len(splits)} chunks")
splits = clean_document_list(splits)
print("10. Documents cleaned")

print("First split preview:", splits[0].page_content[:100] + "...")
print(f"Document1 split into {len(splits)} chunks.")

print("11. About to create Chroma vectorstore")
vectorstore1 = Chroma(collection_name="knowledge1_collection",embedding_function=embeddings_openai, persist_directory="./chroma_knowledge1_db")
print("12. Vectorstore created")

print("13. About to add documents to vectorstore")
try:
    vectorstore1.add_documents(splits)
    print("14. Documents added successfully")
except Exception as e:
    print(f"ERROR adding documents: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("Embedded vectors.")
print("Total Vectors:", vectorstore1._collection.count())
retriever = vectorstore1.as_retriever()

print("15. About to pull prompt from hub")
prompt = hub.pull("rlm/rag-prompt")
print("16. Prompt pulled")

# LLM
print("17. About to create LLM")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
print("18. LLM created")

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

print("19. About to invoke RAG chain")
# Question
answer = rag_chain.invoke("What is Task Decomposition?")
print("20. RAG chain completed")
print("Answer:", answer)
