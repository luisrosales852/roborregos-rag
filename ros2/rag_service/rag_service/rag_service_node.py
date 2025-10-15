#!/usr/bin/env python3
"""
ROS2 RAG Service Node

This node provides a ROS2 service interface for the RAG (Retrieval-Augmented Generation) system.
It wraps the existing RAG implementation from the langchain module and exposes it as a ROS2 service.

Service: /rag_query (rag_service/srv/RAGQuery)
    - Request: question (string)
    - Response: answer (string), success (bool), error_message (string),
                response_time (float32), from_cache (bool)

Author: Generated for RoboRregos RAG Application
"""

import rclpy
from rclpy.node import Node
from rag_interfaces.srv import RAGQuery
import sys
import os
import time
from pathlib import Path
import redis

# Get the ROS2 package directory
SCRIPT_DIR = Path(__file__).resolve().parent
if Path('/ros2_ws/src/rag_service').exists():
    PACKAGE_ROOT = Path('/ros2_ws/src/rag_service')
else:
    PACKAGE_ROOT = SCRIPT_DIR.parent


class RAGServiceNode(Node):
    """
    ROS2 Node that provides RAG query service.

    This node initializes the RAG system from the langchain module and provides
    a service interface for querying the system. It handles both cached and
    non-cached queries, tracks response times, and provides error handling.
    """

    def __init__(self):
        """Initialize the RAG service node and set up the RAG system."""
        super().__init__('rag_service_node')

        # Declare ROS2 parameters
        self.declare_parameter('knowledge1_pdf', 'data/pdfs/knowledge1.pdf')
        self.declare_parameter('knowledge2_pdf', 'data/pdfs/knowledge2.pdf')
        self.declare_parameter('vector_db1_path', 'data/vector_dbs/chroma_knowledge1_db')
        self.declare_parameter('vector_db2_path', 'data/vector_dbs/chroma_knowledge2_db')
        self.declare_parameter('chunk_size', 400)
        self.declare_parameter('chunk_overlap', 50)
        self.declare_parameter('retrieval_k', 3)

        self.get_logger().info('Initializing RAG Service Node...')

        try:
            # Import the RAG system components
            self._import_rag_components()

            # Initialize the RAG system
            self._initialize_rag_system()

            # Create the service
            self.srv = self.create_service(
                RAGQuery,
                'rag_query',
                self.handle_rag_query
            )

            self.get_logger().info('RAG Service Node initialized successfully')
            self.get_logger().info('Service available at: /rag_query')

        except Exception as e:
            self.get_logger().error(f'Failed to initialize RAG Service Node: {str(e)}')
            raise

    def _import_rag_components(self):
        """Import necessary components from the langchain module."""
        try:
            # Import environment loading
            from dotenv import load_dotenv
            # Load .env from package root
            env_path = PACKAGE_ROOT / '.env'
            load_dotenv(dotenv_path=env_path)

            # Get environment variables
            self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            self.get_logger().info(f'Redis URL: {self.redis_url}')

            # Import all necessary modules from main.py
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
            from pydantic import BaseModel, Field
            from rag_service.caching import RAGCacheManager  # Import from package caching.py
            from langchain.globals import set_llm_cache
            from langchain_community.cache import RedisCache

            # Store imports as instance variables for later use
            self.modules = {
                're': re,
                'RecursiveCharacterTextSplitter': RecursiveCharacterTextSplitter,
                'StrOutputParser': StrOutputParser,
                'RunnablePassthrough': RunnablePassthrough,
                'RunnableLambda': RunnableLambda,
                'ChatOpenAI': ChatOpenAI,
                'OpenAIEmbeddings': OpenAIEmbeddings,
                'Chroma': Chroma,
                'PyPDFLoader': PyPDFLoader,
                'CharacterTextSplitter': CharacterTextSplitter,
                'Document': Document,
                'ChatPromptTemplate': ChatPromptTemplate,
                'dumps': dumps,
                'loads': loads,
                'itemgetter': itemgetter,
                'BM25Retriever': BM25Retriever,
                'datetime': datetime,
                'date': date,
                'BaseModel': BaseModel,
                'Field': Field,
                'RAGCacheManager': RAGCacheManager,
                'set_llm_cache': set_llm_cache,
                'RedisCache': RedisCache
            }

            self.get_logger().info('Successfully imported RAG components')

        except ImportError as e:
            self.get_logger().error(f'Failed to import RAG components: {str(e)}')
            raise

    def _initialize_rag_system(self):
        """Initialize the RAG system with all necessary components."""
        try:
            redis_client = redis.Redis.from_url(self.redis_url, decode_responses=True)
            # Set up LLM cache

            self.modules['set_llm_cache'](
                self.modules['RedisCache'](redis_=redis_client)
            )

            # Initialize cache manager
            self.cache_manager = self.modules['RAGCacheManager'](
                redis_url=self.redis_url,
                cache_prefix="rag_cache",
                max_qa_pairs=10000
            )

            # Get current time and date
            self.current_time = self.modules['datetime'].now().time()
            self.current_date = self.modules['date'].today()

            # Initialize LLM
            self.llm = self.modules['ChatOpenAI'](
                model_name="gpt-4o-mini",
                temperature=0
            )


            # Load PDF documents from package root
            pdf1_path = PACKAGE_ROOT / "data/pdfs/knowledge1.pdf"
            pdf2_path = PACKAGE_ROOT / "data/pdfs/knowledge2.pdf"
            self.get_logger().info(f'Loading PDF 1: {pdf1_path}')
            self.get_logger().info(f'Loading PDF 2: {pdf2_path}')

            loader1 = self.modules['PyPDFLoader'](str(pdf1_path))
            loader2 = self.modules['PyPDFLoader'](str(pdf2_path))
            docs1 = loader1.load()
            docs2 = loader2.load()

            # Initialize embeddings
            embeddings_openai = self.modules['OpenAIEmbeddings'](
                model="text-embedding-3-large"
            )
            embeddings_openai = self.cache_manager.setup_cached_embeddings(embeddings_openai)

            # Split documents
            chunk_size = self.get_parameter('chunk_size').value
            chunk_overlap = self.get_parameter('chunk_overlap').value

            text_splitter = self.modules['RecursiveCharacterTextSplitter'](
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            splits1 = text_splitter.split_documents(docs1)
            splits1 = self._clean_document_list(splits1)

            text_splitter2 = self.modules['CharacterTextSplitter'](
                separator="---",
                chunk_size=1000,
                chunk_overlap=0
            )
            splits2 = text_splitter2.split_documents(docs2)
            splits2 = self._clean_document_list(splits2)

            self.get_logger().info(f'Document 1 split into {len(splits1)} chunks')
            self.get_logger().info(f'Document 2 split into {len(splits2)} chunks')

            # Initialize vector stores (paths relative to package root)
            vector_db1_path = self.get_parameter('vector_db1_path').value
            vector_db2_path = self.get_parameter('vector_db2_path').value

            # Make paths absolute from package root
            vector_db1_path = str((PACKAGE_ROOT / vector_db1_path).resolve())
            vector_db2_path = str((PACKAGE_ROOT / vector_db2_path).resolve())

            self.vectorstore1 = self.modules['Chroma'](
                collection_name="knowledge1_collection",
                embedding_function=embeddings_openai,
                persist_directory=vector_db1_path
            )

            self.vectorstore2 = self.modules['Chroma'](
                collection_name="knowledge2_collection",
                embedding_function=embeddings_openai,
                persist_directory=vector_db2_path
            )

            # Check if we need to populate vector stores
            if not os.path.exists(vector_db1_path):
                from uuid import uuid4
                uuids1 = [str(uuid4()) for _ in range(len(splits1))]
                self.vectorstore1.add_documents(documents=splits1, ids=uuids1)
                self.get_logger().info('Populated vector store 1')

            if not os.path.exists(vector_db2_path):
                from uuid import uuid4
                uuids2 = [str(uuid4()) for _ in range(len(splits2))]
                self.vectorstore2.add_documents(documents=splits2, ids=uuids2)
                self.get_logger().info('Populated vector store 2')

            # Create retrievers
            retrieval_k = self.get_parameter('retrieval_k').value
            self.retriever1 = self.vectorstore1.as_retriever(
                search_kwargs={"k": retrieval_k}
            )
            self.retriever2 = self.vectorstore2.as_retriever(
                search_kwargs={"k": retrieval_k}
            )

            # Create BM25 retrievers
            self.bm25_retriever1 = self.modules['BM25Retriever'].from_documents(
                splits1, k=retrieval_k
            )
            self.bm25_retriever2 = self.modules['BM25Retriever'].from_documents(
                splits2, k=retrieval_k
            )

            # Initialize graders and prompts
            self._initialize_graders_and_prompts()

            self.get_logger().info('RAG system initialized successfully')

        except Exception as e:
            self.get_logger().error(f'Failed to initialize RAG system: {str(e)}')
            raise

    def _initialize_graders_and_prompts(self):
        """Initialize document graders and prompt templates."""
        # Define Pydantic models for structured outputs
        class GradeDocuments(self.modules['BaseModel']):
            """Binary score for relevance check on retrieved documents."""
            binary_score: str = self.modules['Field'](
                description="Documents are relevant to the question, 'yes' or 'no'"
            )

        class TaskDistinction(self.modules['BaseModel']):
            """Binary score for task distinction that needs to be ran"""
            dynamic_skill: str = self.modules['Field'](
                description="Will you need to consult the vector database in order to get the information you need." \
                " The vector database is filled with information like how I met my friend el inmortal . Answer exclusively with "
                " 'no'. If I reference in any way a character named inmortal please do put 'yes' in this part. I also talk a little bit about myself in this vector store" \
                "My name is Luis ALvaro Rosales Salazar so if I mention Luis put yes, also even if just the word inmortal is present." \
                "I also have a second vector store, this vector store talks about the products that Reflex , a company specializing in selling customers and enterprises" \
                "construction goods like glue or cement, has. If at any point I mention reflex or anything pertaining to construction please answer yes since I am going to" \
                "need to consult this vector database"
            )
            static_skill: str = self.modules['Field'](
                description="Will you use any of these skills, current time or current date. Answer with 'yes' or 'no' and only answer yes if youre very sure that thats what the user wants."
            )

        class VectorStoreDistinction(self.modules['BaseModel']):
            """Binary score for vector store distinction between these different vector stores."""
            inmortalVector: str = self.modules['Field'](
                description="This vector database has information pertaining to a character named El inmortal and his various exploits as well as" \
                "some personal information about the author Luis Alvaro Rosales Salazar. Only answer with yes or no if youre absolutely sure you need to consult this vector database. Be very sure of your answer as for the values in this " \
                "output object will be mutually exclusive."
            )
            reflexVector: str = self.modules['Field'](
                description="This vector database has information pertaining to reflex products and inventory. Reflex is a company specializing in making construction goods and it also " \
                "sells customer goods like glue. Answer exclusively with yes or no  to the question of do you think we need to search this specific vector store to return the " \
                "appropiate answer to the user. Be very sure of your answer since you can only consult one vector store. If even the name of reflex is mentioned or if anything pertaining to construction" \
                "is mentioned you should use this vector store"
            )

        # Task distinction grader
        structured_llm_task_dist = self.llm.with_structured_output(TaskDistinction)
        system_task_distinction = """You are an expert at identifying the type of skills or movements an llm must make in order to answer
a users question. You must answer yes or no to each of the questions I present to you and only say yes if youre absolutely sure. If the question ends
up using both """

        task_distinction_prompt = self.modules['ChatPromptTemplate'].from_messages([
            ("system", system_task_distinction),
            ("human", "Question: {question}")
        ])

        self.task_grader = task_distinction_prompt | structured_llm_task_dist

        # Document relevance grader
        structured_llm_grader = self.llm.with_structured_output(GradeDocuments)
        system = """You are a grader assessing relevance of a retrieved document to a user question. \n
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

        grade_prompt = self.modules['ChatPromptTemplate'].from_messages([
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ])

        self.retrieval_grader = grade_prompt | structured_llm_grader

        # Vector store router
        structured_llm_vector = self.llm.with_structured_output(VectorStoreDistinction)
        system_vector_dist = """You are an expert at routing questions to the appropriate vector database.
You have access to two vector databases and must choose exactly one based on the question content.
Answer with 'yes' or 'no' for each database, ensuring only one gets 'yes'."""

        vectorDistPrompt = self.modules['ChatPromptTemplate'].from_messages([
            ("system", system_vector_dist),
            ("human", "Question: {question}")
        ])

        self.vector_store_grader = vectorDistPrompt | structured_llm_vector

        # BM25 query generation
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

        prompt_bm25 = self.modules['ChatPromptTemplate'].from_template(template_bm25)
        self.generate_bm25_query = prompt_bm25 | self.modules['ChatOpenAI'](temperature=0) | self.modules['StrOutputParser']()

        # Step Back Prompting for query generation
        template = """Eres un asistente de modelo de lenguaje de IA. Tu tarea es generar 3
versiones diferentes de la pregunta del usuario para recuperar documentos relevantes de una base
de datos vectorial. Al generar múltiples perspectivas sobre la pregunta del usuario, tu objetivo es ayudar
al usuario a superar algunas de las limitaciones de la búsqueda por similitud basada en distancia.
Proporciona estas preguntas alternativas separadas por saltos de línea. Pregunta original: {question}"""

        prompt_perspectives = self.modules['ChatPromptTemplate'].from_template(template)
        self.generate_queries = (
            prompt_perspectives
            | self.modules['ChatOpenAI'](temperature=0)
            | self.modules['StrOutputParser']()
            | (lambda x: x.split("\n"))
        )

        # Final RAG prompt
        template = """Responde a la siguiente pregunta usando el contexto seleccionado

        {context}

        Pregunta: {question}
        """
        self.prompt = self.modules['ChatPromptTemplate'].from_template(template)

    def _clean_pdf_text(self, text):
        """Clean PDF text by removing extra whitespace."""
        cleaned_text = self.modules['re'].sub(r'\s+', ' ', text)
        cleaned_text = cleaned_text.strip()
        return cleaned_text

    def _clean_document_list(self, docs):
        """Clean a list of Document objects, preserving metadata."""
        cleaned_docs = []
        for doc in docs:
            cleaned_content = self._clean_pdf_text(doc.page_content)
            cleaned_doc = self.modules['Document'](
                page_content=cleaned_content,
                metadata=doc.metadata
            )
            cleaned_docs.append(cleaned_doc)
        return cleaned_docs

    def _format_docs(self, docs):
        """Format documents for context."""
        return "\n\n".join([doc.page_content for doc in docs])

    def _reciprocal_rank_fusion(self, results, k=60):
        """Reciprocal rank fusion for combining multiple ranked document lists."""
        fused_scores = {}

        for id_list, docs in enumerate(results):
            for rank, doc in enumerate(docs):
                doc_str = self.modules['dumps'](doc)
                score_docs = 1/(rank+k)
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                fused_scores[doc_str] += score_docs

        reranked_results = [
            (self.modules['loads'](doc), score)
            for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]

        return reranked_results

    def _grade_docs_final(self, docs, question):
        """Grade and filter documents using LLM structured output."""
        filtered_docs = []

        for doc in docs:
            score = self.retrieval_grader.invoke({
                "question": question,
                "document": doc.page_content
            })

            if score.binary_score == "yes":
                self.get_logger().info("Document is relevant")
                filtered_docs.append(doc)
            else:
                self.get_logger().info("Document is not relevant")

        return filtered_docs

    def _hybrid_retrieval(self, query_input, vector_store_choice):
        """Perform hybrid retrieval using both BM25 and vector search."""
        question = query_input["question"]

        if vector_store_choice == "vector1":
            retriever = self.retriever1
            bm25_retriever = self.bm25_retriever1
        else:
            retriever = self.retriever2
            bm25_retriever = self.bm25_retriever2

        # BM25 retrieval
        bm25_query = self.generate_bm25_query.invoke({"question": question})
        self.get_logger().info(f'BM25 query: {bm25_query}')
        bm25_docs = bm25_retriever.invoke(bm25_query)

        # Vector search - use a lambda to log the generated questions
        def debug_queries(queries):
            self.get_logger().info(f'Generated {len(queries)} question variations:')
            for i, q in enumerate(queries, 1):
                self.get_logger().info(f'  {i}. {q}')
            return queries

        retrieval_chain = self.generate_queries | self.modules['RunnableLambda'](debug_queries) | retriever.map()
        vector_docs = retrieval_chain.invoke(query_input)
        vector_docs = [doc for doc_list in vector_docs for doc in doc_list]
        self.get_logger().info(f'Vector search retrieved {len(vector_docs)} documents')

        # Combine both sets of documents
        self.get_logger().info(f'BM25 docs: {len(bm25_docs)}, Vector docs: {len(vector_docs)}')
        combined_docs = [bm25_docs, vector_docs]
        scored_docs = self._reciprocal_rank_fusion(combined_docs)
        self.get_logger().info(f'After RRF fusion: {len(scored_docs)} unique documents')

        all_docs = [doc for doc, score in scored_docs]
        self.get_logger().info(f'Grading {len(all_docs)} documents...')
        filtered_docs = self._grade_docs_final(all_docs, question)
        self.get_logger().info(f'After grading: {len(filtered_docs)} relevant documents')

        return filtered_docs

    def _create_rag_response(self, question, vector_store_choice):
        """Create a RAG response for the given question."""
        def parameterized_hybrid_retrieval(query_input):
            return self._hybrid_retrieval(query_input, vector_store_choice)

        final_rag_chain = (
            {"context": self.modules['RunnableLambda'](parameterized_hybrid_retrieval) | self.modules['RunnableLambda'](self._format_docs),
             "question": self.modules['itemgetter']("question")}
            | self.prompt
            | self.llm
            | self.modules['StrOutputParser']()
        )

        return final_rag_chain.invoke({"question": question})

    def _handle_static_skills(self, question):
        """Handle questions that require static skills - provide all available info."""
        context = f"""Available static information:
    Current time: {self.current_time}
    Current date: {self.current_date}"""

        static_prompt = self.modules['ChatPromptTemplate'].from_template(
            """Answer the user's question using the provided information.

    {context}

    Question: {question}
    """
        )

        static_chain = static_prompt | self.llm | self.modules['StrOutputParser']()
        return static_chain.invoke({"context": context, "question": question})

    def _route_question(self, question):
        """Route the question to the appropriate handler and return the answer."""
        # Check cache first
        cached_answer = self.cache_manager.get_cached_answer(question)
        if cached_answer:
            self.get_logger().info('Answer retrieved from cache')
            return cached_answer, True

        self.get_logger().info('Analyzing question type...')
        task_result = self.task_grader.invoke({"question": question})
        self.get_logger().info(f'Dynamic skill needed: {task_result.dynamic_skill}')
        self.get_logger().info(f'Static skill needed: {task_result.static_skill}')

        if task_result.dynamic_skill == "yes":
            self.get_logger().info('Routing to RAG chain')
            vector_result = self.vector_store_grader.invoke({"question": question})
            self.get_logger().info(f'Inmortal vector: {vector_result.inmortalVector}')
            self.get_logger().info(f'Reflex Vector: {vector_result.reflexVector}')

            if vector_result.inmortalVector == "yes":
                self.get_logger().info('Routing to Inmortal Vector Store')
                answer = self._create_rag_response(question, "vector1")
            elif vector_result.reflexVector == "yes":
                self.get_logger().info('Routing to Reflex Vector Store')
                answer = self._create_rag_response(question, "vector2")
            else:
                self.get_logger().info('No specific vector store selected, defaulting to vector store 1')
                answer = self._create_rag_response(question, "vector1")

        elif task_result.static_skill == "yes":
            self.get_logger().info('Routing to static skills')
            answer = self._handle_static_skills(question)
        else:
            self.get_logger().info('Routing to basic LLM')
            basic_prompt = self.modules['ChatPromptTemplate'].from_template("Answer this question: {question}")
            basic_chain = basic_prompt | self.llm | self.modules['StrOutputParser']()
            answer = basic_chain.invoke({"question": question})

        # Cache the answer
        self.cache_manager.cache_qa_pair(question, answer)
        return answer, False

    def handle_rag_query(self, request, response):
        """
        Handle incoming RAG query requests.

        Args:
            request: RAGQuery service request containing the question
            response: RAGQuery service response to be filled

        Returns:
            response: Filled service response
        """
        start_time = time.time()

        self.get_logger().info(f'Received query: {request.question}')

        try:
            # Process the query
            answer, from_cache = self._route_question(request.question)

            # Calculate response time
            response_time = time.time() - start_time

            # Fill the response
            response.answer = answer
            response.success = True
            response.error_message = ""
            response.response_time = response_time
            response.from_cache = from_cache

            self.get_logger().info(f'Query processed successfully in {response_time:.2f}s (cached: {from_cache})')

        except Exception as e:
            response_time = time.time() - start_time
            error_msg = f'Error processing query: {str(e)}'

            self.get_logger().error(error_msg)

            response.answer = ""
            response.success = False
            response.error_message = error_msg
            response.response_time = response_time
            response.from_cache = False

        return response


def main(args=None):
    """Main entry point for the RAG service node."""
    rclpy.init(args=args)

    try:
        rag_service_node = RAGServiceNode()
        rclpy.spin(rag_service_node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error in RAG service node: {str(e)}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
