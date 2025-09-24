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
from datetime import datetime, date
import functions
from langchain_core.pydantic_v1 import BaseModel, Field


#Definir static fact and dynamic fact

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
    vector_docs = [doc for doc_list in vector_docs for doc in doc_list]
    
    # Combine both sets of documents
    combined_docs = [bm25_docs, vector_docs]
    final_docs = reciprocal_rank_fusion(combined_docs)

    final_docs = [doc for doc,scores in final_docs[:6]]
    
    # Use your existing function to get unique documents
    return final_docs

#Function to rank selection. Update pls.
def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
        and an optional parameter k used in the RRF formula """
    
    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    bm25_scores = []
    vector_scores = []


    # Iterate through each list of ranked documents
    for id_list, docs in enumerate(results):
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            score_docs = 1/(rank+k)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += score_docs
            if id_list == 0:
                bm25_scores.append(score_docs)
            else:
                vector_scores.append(score_docs)

    if bm25_scores:
        print(f"Average BM25 RRF Score: {sum(bm25_scores)/len(bm25_scores):.6f}")
    if vector_scores:
        print(f"Average Vector RRF Score: {sum(vector_scores)/len(vector_scores):.6f}")

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results


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

def debug_queries(queries):
    print("=== GENERATED QUESTIONS ===")
    for i, q in enumerate(queries, 1):
        print(f"{i}. {q}")
    print("=========================")
    return queries


current_time = datetime.now().time()
current_date = date.today()

Type_Skills = {
    "static_skill": False,
    "dynamic_skill": False,
}

Skills = {
    "current_time": False,
    "current_date": False,
    "vector_store": False,
},

script_dir = os.path.dirname(os.path.abspath(__file__))
pdf1_path = os.path.join(script_dir, "knowledge1.pdf")

print("Script directory:", script_dir)
loader = PyPDFLoader(pdf1_path)
docs = loader.load()

print(loader)
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

# LLM and states

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )



llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


# Implementando BM25 a mi modelo. Aver si era esto. Viva langchain. 
#Voy a perform BM25 en la pregunta inicial, no en las consiguientes 3 gracias a step back prompting aunque primero si la transformaria.

template_bm25 = """Eres un experto en optimización de consultas para búsqueda BM25. Tu tarea es reescribir la pregunta del usuario para maximizar la efectividad de la búsqueda por palabras clave.

Instrucciones para reescribir:
- Elimina palabras de pregunta (qué, cómo, por qué, cuándo, dónde)
- Elimina artículos, preposiciones y conectores innecesarios
- Convierte verbos a sustantivos cuando sea posible
- Mantén nombres propios, términos técnicos y conceptos específicos
- No inventes palabras que no esten en la pregunta original
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


retrieval_chain = generate_queries | debug_queries | retriever.map()

template = """Responde a la siguiente pregunta usando el contexto seleccionado

{context}

Pregunta: {question}
"""


prompt = ChatPromptTemplate.from_template(template)








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
