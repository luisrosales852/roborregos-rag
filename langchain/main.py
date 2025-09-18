import os
from dotenv import load_dotenv
from uuid import uuid4

load_dotenv()

import re
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from langchain.load import dumps, loads
from operator import itemgetter
from langchain_community.retrievers import BM25Retriever

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





# Implementando BM25 a mi modelo. Aver si era esto. Viva langchain. 
#Voy a perform BM25 en la pregunta inicial, no en las consiguientes 3 gracias a step back prompting aunque primero si la transformaria.

template_bm25 = """Eres un experto en optimización de consultas para búsqueda BM25. Tu tarea es reescribir la pregunta del usuario para maximizar la efectividad de la búsqueda por palabras clave.

Instrucciones para reescribir:
- Elimina palabras de pregunta (qué, cómo, por qué, cuándo, dónde)
- Elimina artículos, preposiciones y conectores innecesarios
- Convierte verbos a sustantivos cuando sea posible
- Mantén nombres propios, términos técnicos y conceptos específicos
- Usa sinónimos adicionales si pueden ayudar a encontrar más documentos relevantes
- Estructura como una lista de términos clave separados por espacios
- Máximo 8-10 palabras en la consulta reescrita

Pregunta original: {question}

Consulta optimizada para BM25:"""
prompt_bm25 = ChatPromptTemplate.from_template(template_bm25)
generate_bm25_query = prompt_bm25 | ChatOpenAI(temperature=0) | StrOutputParser()
bm25_query = generate_bm25_query.invoke({"question": question})
print("BM25 Optimized Query:", bm25_query)
bm25_retriever = BM25Retriever.from_documents(splits, k=3)
results_bm25 = bm25_retriever.invoke(bm25_query)
print("BM25 Results:", results_bm25)




# Step Back Prompting
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

def debug_formatted_prompt(formatted_prompt):
    print("Prompt")
    print(formatted_prompt.messages[0].content)  # For ChatPromptTemplate messages
    print("===============================")
    return formatted_prompt

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def hybrid_retrieval(query_input):
    question = query_input["question"]
    
    # Get BM25 results (you already have this)
    bm25_query = generate_bm25_query.invoke({"question": question})
    bm25_docs = bm25_retriever.invoke(bm25_query)
    
    # Get vector search results
    vector_docs = retrieval_chain.invoke(query_input)
    
    # Combine both sets of documents
    combined_docs = [bm25_docs, vector_docs]
    
    # Use your existing function to get unique documents
    return get_unique_union(combined_docs)

final_rag_chain = (
    {"context": hybrid_retrieval | RunnableLambda(format_docs), 
     "question": itemgetter("question")}
    | prompt
    | RunnableLambda(debug_formatted_prompt)
    | llm
    | StrOutputParser()
)
answer = final_rag_chain.invoke({"question":question})
print(answer)
