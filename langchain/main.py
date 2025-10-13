import os
from dotenv import load_dotenv
from uuid import uuid4

load_dotenv()
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

import re
import redis
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
from pydantic import BaseModel, Field
from caching import RAGCacheManager
from langchain.globals import set_llm_cache
from langchain_community.cache import RedisCache

#Setting llm cache?
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
set_llm_cache(RedisCache(redis_=redis_client))


#Definir static fact and dynamic fact
cache_manager = RAGCacheManager(redis_url=REDIS_URL, cache_prefix="rag_cache", max_qa_pairs=10000)

def handle_static_skills(question):
    """Handle questions that require static skills - provide all available info"""
    
    # Provide all static information available
    context = f"""Available static information:
    Current time: {current_time}
    Current date: {current_date}"""
    
    static_prompt = ChatPromptTemplate.from_template(
        """Answer the user's question using the provided information.

    {context}

    Question: {question}
    """
    )
    
    static_chain = static_prompt | RunnableLambda(debug_formatted_prompt) | llm | StrOutputParser()
    return static_chain.invoke({"context": context, "question": question})


def routeQuestion(question):
    cached_answer = cache_manager.get_cached_answer(question)
    if cached_answer:
        return cached_answer
    
    print("Lets analyse the question type")
    task_result = task_grader.invoke({"question": question})
    print(f"Dynamic skill needed: {task_result.dynamic_skill}")
    print(f"Static skill needed: {task_result.static_skill}")
    
    if task_result.dynamic_skill == "yes":
        print("We route to rag chain")
        print("Now to decide to which vector store do we route this too")
        vector_result = vector_store_grader.invoke({"question": question})
        print(f"Inmortal vector: {vector_result.inmortalVector}")
        print(f"Reflex Vector: {vector_result.reflexVector}")
        if vector_result.inmortalVector == "yes":
            print("Routing to Inmortal Vector Store")
            answer = create_rag_response(question, "vector1")
        elif vector_result.reflexVector == "yes":
            print("Routing to Reflex Vector Store")
            answer = create_rag_response(question, "vector2")
        else:
            print("No specific vector store selected, defaulting to vector store 1")
            answer = create_rag_response(question, "vector1")
        
    elif task_result.static_skill == "yes":
        print("We route to static skills")
        answer = handle_static_skills(question)
    else:
        print("---ROUTING TO BASIC LLM---")
        basic_prompt = ChatPromptTemplate.from_template("Answer this question: {question}")
        basic_chain = basic_prompt | llm | StrOutputParser()
        answer = basic_chain.invoke({"question": question})
    cache_manager.cache_qa_pair(question, answer)
    return answer
    

def debug_formatted_prompt(formatted_prompt):
    print("Prompt")
    print(formatted_prompt.messages[0].content)  # For ChatPromptTemplate messages
    print("===============================")
    return formatted_prompt

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def hybrid_retrieval(query_input, vector_store_choice):
    question = query_input["question"]
    
    if vector_store_choice == "vector1":
        retriever = retriever1
        bm25_retriever = bm25_retriever1
    else:
        retriever = retriever2
        bm25_retriever = bm25_retriever2

    bm25_query = generate_bm25_query.invoke({"question": question})
    print(f"These is the better bm25 query: {bm25_query}")
    bm25_docs = bm25_retriever.invoke(bm25_query)

    print("This is the first bm25 docs")
    print(bm25_docs[0])
    
    # Get vector search results
    retrieval_chain = generate_queries | debug_queries | retriever.map()
    vector_docs = retrieval_chain.invoke(query_input)
    vector_docs = [doc for doc_list in vector_docs for doc in doc_list]
    
    # Combine both sets of documents
    combined_docs = [bm25_docs, vector_docs]
    scored_docs = reciprocal_rank_fusion(combined_docs)

    all_docs = [doc for doc, score in scored_docs]

    filtered_docs = gradeDocsFinal(all_docs)
    
    # Use your existing function to get unique documents
    return filtered_docs

#Function to create rag response
def create_rag_response(question, vector_store_choice):
    def parameterized_hybrid_retrieval(query_input):
        return hybrid_retrieval(query_input, vector_store_choice)
    final_rag_chain = (
        {"context": RunnableLambda(parameterized_hybrid_retrieval) | RunnableLambda(format_docs), 
         "question": itemgetter("question")}
        | prompt
        | RunnableLambda(debug_formatted_prompt)
        | llm
        | StrOutputParser()
    )

    return final_rag_chain.invoke({"question": question})

#Function to rank selection. Update pls. May not be needed but we will see. Hmmmm.
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


Skills = {
    "current_time": False,
    "current_date": False,
},

script_dir = os.path.dirname(os.path.abspath(__file__))
pdf1_path = os.path.join(script_dir, "knowledge1.pdf")
pdf2_path = os.path.join(script_dir, "knowledge2.pdf")

print("Script directory:", script_dir)
loader1 = PyPDFLoader(pdf1_path)
loader2 = PyPDFLoader(pdf2_path)
docs1 = loader1.load()
docs2 = loader2.load()

print(loader1)
embeddings_openai = OpenAIEmbeddings(model="text-embedding-3-large")
embeddings_openai = cache_manager.setup_cached_embeddings(embeddings_openai)


# Split first document
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
splits1 = text_splitter.split_documents(docs1)
splits1 = clean_document_list(splits1)

print(splits1[0])
print(f"Document1 split into {len(splits1)} chunks.")

# Split second document
text_splitter2 = CharacterTextSplitter(separator="---",chunk_size=1000, chunk_overlap=0)
splits2 = text_splitter2.split_documents(docs2)
splits2 = clean_document_list(splits2)

print(splits2[0])
print(f"Document 2 split into {len(splits2)} chunks")


uuids1 = [str(uuid4()) for _ in range(len(splits1))]
uuids2 = [str(uuid4()) for _ in range(len(splits2))]
existing_vectorstore = os.path.exists("./chroma_knowledge1_db")
existing_vectorstore2 = os.path.exists("./chroma_knowledge2_db")

vectorstore1 = Chroma(collection_name="knowledge1_collection",embedding_function=embeddings_openai, persist_directory="./chroma_knowledge1_db")
vectorstore2 = Chroma(collection_name="knowledge2_collection", embedding_function=embeddings_openai, persist_directory="./chroma_knowledge2_db")

if(existing_vectorstore == False):
    vectorstore1.add_documents(documents=splits1, ids=uuids1)
    print("Added documents to vectorstore")

if(existing_vectorstore2 == False):
    vectorstore2.add_documents(documents=splits2, ids=uuids2)



retriever1 = vectorstore1.as_retriever(search_kwargs={"k": 3})
retriever2 = vectorstore2.as_retriever(search_kwargs={"k": 3})

question = input("Whats your question?: ")

# LLM and states

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class TaskDistinction(BaseModel):
    """Binary score for task distinction that needs to be ran"""

    dynamic_skill: str = Field(
        description="Will you need to consult the vector database in order to get the information you need." \
        " The vector database is filled with information like how I met my friend el inmortal . Answer exclusively with "
        " 'no'. If I reference in any way a character named inmortal please do put 'yes' in this part. Even if just the word inmortal is present." \
        "I also have a second vector store, this vector store talks about the products that Reflex , a company specializing in selling customers and enterprises" \
        "construction goods like glue or cement, has. If at any point I mention reflex or anything pertaining to construction please answer yes since I am going to" \
        "need to consult this vector database"
    )

    static_skill: str = Field(
        description="Will you use any of these skills, current time or current date. Answer with 'yes' or 'no' and only answer yes if youre very sure that thats what the user wants."
    )



class VectorStoreDistinction(BaseModel):
    """Binary score for vector store distinction between these different vector stores."""

    inmortalVector: str = Field(
        description="This vector database has information pertaining to a character named El inmortal and his various exploits as well as" \
        "some personal information about the author Luis Alvaro Rosales Salazar. Only answer with yes or no if youre absolutely sure you need to consult this vector database. Be very sure of your answer as for the values in this " \
        "output object will be mutually exclusive."
    )

    reflexVector: str = Field(
        description="This vector database has information pertaining to reflex products and inventory. Reflex is a company specializing in making construction goods and it also " \
        "sells customer goods like glue. Answer exclusively with yes or no  to the question of do you think we need to search this specific vector store to return the " \
        "appropiate answer to the user. Be very sure of your answer since you can only consult one vector store. If even the name of reflex is mentioned or if anything pertaining to construction" \
        "is mentioned you should use this vector store"
    )

# More functions?.

def grade_documents(docs):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    documents = docs

    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {"documents": filtered_docs, "question": question}


llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

structured_llm_task_dist = llm.with_structured_output(TaskDistinction)

system_task_distinction = """You are an expert at identifying the type of skills or movements an llm must make in order to answer 
a users question. You must answer yes or no to each of the questions I present to you and only say yes if youre absolutely sure. If the question ends
up using both """

task_distinction_prompt = ChatPromptTemplate.from_messages([
    ("system", system_task_distinction),
    ("human", "Question: {question}")
])

task_grader = task_distinction_prompt | structured_llm_task_dist

#Define structured llm Grader
structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader

#Structured llm to route to the appropiate vector store
structured_llm_vector = llm.with_structured_output(VectorStoreDistinction)

system_vector_dist = """You are an expert at routing questions to the appropriate vector database. 
You have access to two vector databases and must choose exactly one based on the question content. 
Answer with 'yes' or 'no' for each database, ensuring only one gets 'yes'."""

vectorDistPrompt = ChatPromptTemplate.from_messages([
    ("system", system_vector_dist),
    ("human", "Question: {question}")
])

vector_store_grader = vectorDistPrompt | structured_llm_vector

def gradeDocsFinal(docs):
    """Grade and filter documents using LLM structured output"""
    print("---FILTERING DOCUMENTS WITH LLM GRADER---")
    filtered_docs = []
    
    for doc in docs:
        score = retrieval_grader.invoke({
            "question": question, 
            "document": doc.page_content
        })
        
        if score.binary_score == "yes":
            print(f"---DOCUMENT RELEVANT---")
            filtered_docs.append(doc)
        else:
            print(f"---DOCUMENT NOT RELEVANT---")
    
    print(f"Filtered {len(filtered_docs)} out of {len(docs)} documents")
    return filtered_docs


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



prompt_bm25 = ChatPromptTemplate.from_template(template_bm25)
generate_bm25_query = prompt_bm25 | ChatOpenAI(temperature=0) | StrOutputParser()
bm25_retriever1 = BM25Retriever.from_documents(splits1, k=3)
bm25_retriever2 = BM25Retriever.from_documents(splits2, k=3)

template = """Responde a la siguiente pregunta usando el contexto seleccionado

{context}

Pregunta: {question}
"""


prompt = ChatPromptTemplate.from_template(template)

answer = routeQuestion(question)
print(answer)
