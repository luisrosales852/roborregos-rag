import os
from dotenv import load_dotenv
from uuid import uuid4

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
from datetime import datetime

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