import os
from dotenv import load_dotenv
from uuid import uuid4

load_dotenv()


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
from langchain.prompts import ChatPromptTemplate
from langchain.load import dumps, loads
from operator import itemgetter

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

script_dir = os.path.dirname(os.path.abspath(__file__))
pdf1_path = os.path.join(script_dir, "knowledge1.pdf")

print("Script directory:", script_dir)
loader = PyPDFLoader(pdf1_path)
docs = loader.load()

embeddings_openai = OpenAIEmbeddings(model="text-embedding-3-large")

# Split first document
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
splits = text_splitter.split_documents(docs)
splits = clean_document_list(splits)

print(splits[0])
print(f"Document1 split into {len(splits)} chunks.")

uuids = [str(uuid4()) for _ in range(len(splits))]
existing_vectorstore = os.path.exists("./chroma_knowledge1_db")

vectorstore1 = Chroma(collection_name="knowledge1_collection",embedding_function=embeddings_openai, persist_directory="./chroma_knowledge1_db")

if(existing_vectorstore == False):
    vectorstore1.add_documents(documents=splits, ids=uuids)
    print("Added documents to vectorstore")


retriever = vectorstore1.as_retriever(search_kwargs={"k": 3})

question = input("Whats your question?: ")

# Multi Query: Different Perspectives
template = """Eres un asistente de modelo de lenguaje de IA. Tu tarea es generar 3 
versiones diferentes de la pregunta del usuario para recuperar documentos relevantes de una base 
de datos vectorial. Al generar múltiples perspectivas sobre la pregunta del usuario, tu objetivo es ayudar
al usuario a superar algunas de las limitaciones de la búsqueda por similitud basada en distancia.
Proporciona estas preguntas alternativas separadas por saltos de línea. Pregunta original: {question}"""
prompt_perspectives = ChatPromptTemplate.from_template(template)

generate_queries = (
    prompt_perspectives 
    | ChatOpenAI(temperature=0) 
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)

def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]

def debug_queries(queries):
    print("=== GENERATED QUESTIONS ===")
    for i, q in enumerate(queries, 1):
        print(f"{i}. {q}")
    print("=========================")
    return queries

retrieval_chain = generate_queries | debug_queries | retriever.map() | get_unique_union

template = """Responde a la siguiente pregunta usando el contexto seleccionado

{context}

Pregunta: {question}
"""


prompt = ChatPromptTemplate.from_template(template)

# LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)



final_rag_chain = (
    {"context": retrieval_chain, 
     "question": itemgetter("question")} 
    | prompt
    | llm
    | StrOutputParser()
)

final_inputs = {
    "context": [doc.page_content for doc in retrieval_chain.invoke({"question": question})],
    "question": question
}

print("Final inputs:", final_inputs)

answer = final_rag_chain.invoke({"question":question})
print(answer)
