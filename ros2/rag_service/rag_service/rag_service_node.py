

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
        self.declare_parameter('chunk_overlap', 100)
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
            # Load .env from ros2 folder (parent of package root)
            env_path = PACKAGE_ROOT.parent / '.env'
            load_dotenv(dotenv_path=env_path)

            # Get environment variables
            self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

            # CHATGPT/OPENAI CONFIGURATION (ACTIVE)
            self.openai_api_key = os.getenv("OPENAI_API_KEY")

            # self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            # self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

            self.get_logger().info(f'Redis URL: {self.redis_url}')
            # self.get_logger().info(f'Ollama Base URL: {self.ollama_base_url}')
            # self.get_logger().info(f'Ollama Model: {self.ollama_model}')
            self.get_logger().info('Using ChatGPT (OpenAI) for LLM and embeddings')

            # Import all necessary modules from main.py
            import re
            from langchain import hub
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.runnables import RunnablePassthrough, RunnableLambda
            # OPENAI IMPORTS (ACTIVE)
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings

            # from langchain_ollama import ChatOllama, OllamaEmbeddings
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

            # Initialize LLM with ChatGPT/OpenAI (ACTIVE)
            self.llm = self.modules['ChatOpenAI'](
                model_name="gpt-4o-mini",
                temperature=0
            )

            # self.llm = self.modules['ChatOllama'](
            #     model=self.ollama_model,  # gpt-oss:20b or llama3.1:8b
            #     base_url=self.ollama_base_url,
            #     temperature=0
            # num_gpu=<layers> to limit GPU usage
            # )


            pdf1_path = PACKAGE_ROOT / "data/pdfs/knowledge1.pdf"
            pdf2_path = PACKAGE_ROOT / "data/pdfs/knowledge2.pdf"
            self.get_logger().info(f'Loading PDF 1: {pdf1_path}')
            self.get_logger().info(f'Loading PDF 2: {pdf2_path}')

            loader1 = self.modules['PyPDFLoader'](str(pdf1_path))
            loader2 = self.modules['PyPDFLoader'](str(pdf2_path))
            docs1 = loader1.load()
            docs2 = loader2.load()

            embeddings_openai = self.modules['OpenAIEmbeddings'](
                model="text-embedding-3-large"
            )
            embeddings_openai = self.cache_manager.setup_cached_embeddings(embeddings_openai)

   
            # embeddings_ollama = self.modules['OllamaEmbeddings'](
            #     model=self.ollama_model,  
            #     base_url=self.ollama_base_url
            # )
            # embeddings_ollama = self.cache_manager.setup_cached_embeddings(embeddings_ollama)

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
                chunk_size=2000,
                chunk_overlap=0
            )
            splits2 = text_splitter2.split_documents(docs2)
            splits2 = self._clean_document_list(splits2)

            self.get_logger().info(f'Document 1 split into {len(splits1)} chunks')
            self.get_logger().info(f'Document 2 split into {len(splits2)} chunks')

            vector_db1_path = self.get_parameter('vector_db1_path').value
            vector_db2_path = self.get_parameter('vector_db2_path').value

            vector_db1_path = str((PACKAGE_ROOT / vector_db1_path).resolve())
            vector_db2_path = str((PACKAGE_ROOT / vector_db2_path).resolve())

            self.vectorstore1 = self.modules['Chroma'](
                collection_name="knowledge1_collection",
                embedding_function=embeddings_openai,  # Change to embeddings_ollama for Ollama
                persist_directory=vector_db1_path
            )

            self.vectorstore2 = self.modules['Chroma'](
                collection_name="knowledge2_collection",
                embedding_function=embeddings_openai,  # Change to embeddings_ollama for Ollama
                persist_directory=vector_db2_path
            )

            if self.vectorstore1._collection.count() == 0:
                from uuid import uuid4
                uuids1 = [str(uuid4()) for _ in range(len(splits1))]
                self.vectorstore1.add_documents(documents=splits1, ids=uuids1)
                self.get_logger().info('Populated vector store 1')

            if self.vectorstore2._collection.count() == 0:
                from uuid import uuid4
                uuids2 = [str(uuid4()) for _ in range(len(splits2))]
                self.vectorstore2.add_documents(documents=splits2, ids=uuids2)
                self.get_logger().info('Populated vector store 2')

            retrieval_k = self.get_parameter('retrieval_k').value
            self.retriever1 = self.vectorstore1.as_retriever(
                search_kwargs={"k": retrieval_k}
            )
            self.retriever2 = self.vectorstore2.as_retriever(
                search_kwargs={"k": retrieval_k}
            )

            self.bm25_retriever1 = self.modules['BM25Retriever'].from_documents(
                splits1, k=retrieval_k
            )
            self.bm25_retriever2 = self.modules['BM25Retriever'].from_documents(
                splits2, k=retrieval_k
            )

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
                description="Los documentos son relevantes a la pregunta, 'si' o 'no'"
            )

        class TaskDistinction(self.modules['BaseModel']):
            """Binary score for task distinction that needs to be ran"""
            dynamic_skill: str = self.modules['Field'](
                description="¿La pregunta menciona o requiere información sobre cualquiera de estos temas: "
                "1. 'El inmortal' o Luis Alvaro Rosales Salazar "
                "2. Reflex, cemento, pegamento, o productos de construcción? "
                "Si la pregunta menciona CUALQUIERA de estas palabras clave: inmortal, Luis, Alvaro, Rosales, Salazar, Reflex, cemento, pegamento, construcción, productos - responde 'si'. "
                "IMPORTANTE: Si detectas las palabras 'inmortal', 'Luis', 'Reflex' o 'reflex' en la pregunta, SIEMPRE responde 'si'. " \
                "Tambien responde que si si es que piensas que el contexto de la pregunta va por el rumbo de esas palabras clave o tenga algo que ver."
                "Solo responde 'no' si la pregunta es sobre matemáticas, conversación general, o temas completamente diferentes. "
                "Responde SOLO 'si' o 'no'."
            )
            static_skill: str = self.modules['Field'](
                description="¿La pregunta requiere información de hora o fecha actual? "
                "Responde SOLO 'si' o 'no'. "
                "Solo responde 'si' si el usuario explícitamente pregunta por la hora o fecha actual."
            )

        class VectorStoreDistinction(self.modules['BaseModel']):
            """Binary score for vector store distinction between these different vector stores."""
            inmortalVector: str = self.modules['Field'](
                description="¿La pregunta menciona 'inmortal' (con mayúscula o minúscula) o 'Luis'? "
                "IMPORTANTE: Si encuentras las palabras 'inmortal', 'Luis', 'Alvaro', 'Rosales' o 'Salazar' en CUALQUIER parte de la pregunta, SIEMPRE responde 'si'. "
                "También incluye variantes como: 'el inmortal', 'immortal', 'Luis Alvaro', etc.  Tambien cualquier cosa que tenga algo que ver con esto."
                "De lo contrario responde 'no'. "
                "Responde SOLO 'si' o 'no'."
            )
            reflexVector: str = self.modules['Field'](
                description="¿La pregunta menciona 'Reflex' (con mayúscula o minúscula) o productos de construcción? "
                "IMPORTANTE: Si encuentras la palabra 'Reflex' o 'reflex' en CUALQUIER parte de la pregunta, SIEMPRE responde 'si'. "
                "También responde 'si' si detectas: cemento, pegamento, construcción, productos, mortero, adhesivo. Tambien cualquier cosa que tenga que ver con construccion"
                "De lo contrario responde 'no'. "
                "Responde SOLO 'si' o 'no'."
            )

        # Task distinction grader
        structured_llm_task_dist = self.llm.with_structured_output(TaskDistinction)
        system_task_distinction = """Eres un experto en identificar el tipo de habilidades o movimientos que un LLM debe hacer para responder
la pregunta de un usuario. IMPORTANTE: Solo puedes responder EXCLUSIVAMENTE con las palabras 'si' o 'no'. No proporciones explicaciones,
no uses otras palabras. SOLO 'si' o 'no' para cada campo."""

        task_distinction_prompt = self.modules['ChatPromptTemplate'].from_messages([
            ("system", system_task_distinction),
            ("human", "Pregunta: {question}")
        ])

        self.task_grader = task_distinction_prompt | structured_llm_task_dist

        # Document relevance grader
        structured_llm_grader = self.llm.with_structured_output(GradeDocuments)
        system = """Eres un evaluador que determina la relevancia de un documento recuperado a la pregunta de un usuario. \n
    Si crees que el documento sera util para responder la pregunta del usuario responde con si, si crees que no sera util responde con un no. Se leniente y trata de responder si mas que no
    No importa que el documento no tenga todo, solo importa que tenga una parte de la pregunta o que tenga algo que ver. \n
    IMPORTANTE: Solo puedes responder EXCLUSIVAMENTE con 'si' o 'no'. No proporciones explicaciones. SOLO 'si' o 'no'."""

        grade_prompt = self.modules['ChatPromptTemplate'].from_messages([
            ("system", system),
            ("human", "Documento recuperado: \n\n {document} \n\n Pregunta del usuario: {question}"),
        ])

        self.retrieval_grader = grade_prompt | structured_llm_grader

        # Vector store router
        structured_llm_vector = self.llm.with_structured_output(VectorStoreDistinction)
        system_vector_dist = """Eres un experto en dirigir preguntas a la base de datos vectorial apropiada.
Tienes acceso a dos bases de datos vectoriales y debes elegir exactamente una basándote en el contenido de la pregunta.
IMPORTANTE: Solo puedes responder EXCLUSIVAMENTE con 'si' o 'no' para cada campo. No proporciones explicaciones. SOLO 'si' o 'no'."""

        vectorDistPrompt = self.modules['ChatPromptTemplate'].from_messages([
            ("system", system_vector_dist),
            ("human", "Pregunta: {question}")
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
        self.generate_bm25_query = prompt_bm25 | self.llm | self.modules['StrOutputParser']()

        # Query Improvement for RAG Efficiency
        template_improvement = """Eres un experto en mejorar las preguntas para poder perform rag en esa pregunta, no pierdas el contexto ni las palabras clave 
        pero si mejoralo si es necesario. Esta es la pregunta original, no la modifiques mucho y si ves una palabra que piensas que es otra cosa no le muevas, lo ultimo que quiero
         es que cambies el contexto de la pregunta, especialmente si ves la palabra inmortal no lo cambies y no insinues nada, es un personaje en una base
         de datos vectorial. Tu existes para evitar principalmente errores de ortografia o de logica graves:{question}

Devuelve la pregunta mejorada"""

        prompt_improvement = self.modules['ChatPromptTemplate'].from_template(template_improvement)
        self.improve_query = prompt_improvement | self.llm | self.modules['StrOutputParser']()

        # Step Back Prompting for query generation
        template = """Eres un asistente de modelo de lenguaje de IA. Tu tarea es generar 3
versiones diferentes de la pregunta del usuario para recuperar documentos relevantes de una base
de datos vectorial. Al generar múltiples perspectivas sobre la pregunta del usuario, tu objetivo es ayudar
al usuario a superar algunas de las limitaciones de la búsqueda por similitud basada en distancia.
Proporciona estas preguntas alternativas separadas por saltos de línea. Pregunta original: {question}"""

        prompt_perspectives = self.modules['ChatPromptTemplate'].from_template(template)
        self.generate_queries = (
            prompt_perspectives
            | self.llm
            | self.modules['StrOutputParser']()
            | (lambda x: [q.strip() for q in x.split("\n") if q.strip()])
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

            if score.binary_score == "si":
                self.get_logger().info("Document is relevant")
                filtered_docs.append(doc)
            else:
                self.get_logger().info("Document is not relevant")

        return filtered_docs

    def _hybrid_retrieval(self, query_input, vector_store_choice):
        """Perform hybrid retrieval using both BM25 and vector search."""
        original_question = query_input["question"]

        # STEP 1: Query Improvement - Improve the original question for better RAG efficiency
        self.get_logger().info(f'Original question: {original_question}')
        question = self.improve_query.invoke({"question": original_question})
        self.get_logger().info(f'Improved question: {question}')

        # Update query_input to use improved question for all downstream processing
        query_input = {"question": question}

        if vector_store_choice == "vector1":
            retriever = self.retriever1
            bm25_retriever = self.bm25_retriever1
        else:
            retriever = self.retriever2
            bm25_retriever = self.bm25_retriever2

        # STEP 2: BM25 retrieval - Uses improved question
        bm25_query = self.generate_bm25_query.invoke({"question": question})
        self.get_logger().info(f'BM25 query: {bm25_query}')
        bm25_docs = bm25_retriever.invoke(bm25_query)

        # STEP 3: Vector search - Uses improved question for step-back prompting
        def debug_queries(queries):
            self.get_logger().info(f'Generated {len(queries)} question variations:')
            for i, q in enumerate(queries, 1):
                self.get_logger().info(f'  {i}. {q}')
            return queries

        retrieval_chain = self.generate_queries | self.modules['RunnableLambda'](debug_queries) | retriever.map()
        vector_docs = retrieval_chain.invoke(query_input)
        vector_docs = [doc for doc_list in vector_docs for doc in doc_list]
        self.get_logger().info(f'Vector search retrieved {len(vector_docs)} documents')

        # STEP 4: Combine both sets of documents using RRF
        self.get_logger().info(f'BM25 docs: {len(bm25_docs)}, Vector docs: {len(vector_docs)}')
        combined_docs = [bm25_docs, vector_docs]
        scored_docs = self._reciprocal_rank_fusion(combined_docs)
        self.get_logger().info(f'After RRF fusion: {len(scored_docs)} unique documents')

        # STEP 5: Grade documents using the improved question
        all_docs = [doc for doc, score in scored_docs]
        self.get_logger().info(f'Grading {len(all_docs)} documents...')
        filtered_docs = self._grade_docs_final(all_docs, question)
        self.get_logger().info(f'After grading: {len(filtered_docs)} relevant documents')

        # Return both filtered docs and improved question for final answer generation
        return {"docs": filtered_docs, "improved_question": question}

    def _create_rag_response(self, question, vector_store_choice):
        """Create a RAG response for the given question."""
        # Perform hybrid retrieval which returns both docs and improved question
        retrieval_result = self._hybrid_retrieval({"question": question}, vector_store_choice)
        filtered_docs = retrieval_result["docs"]
        improved_question = retrieval_result["improved_question"]

        # Format the documents for context
        context = self._format_docs(filtered_docs)

        # Use the improved question with the retrieved documents for final answer
        self.get_logger().info(f'Generating final answer using improved question: {improved_question}')
        final_answer = (
            self.prompt
            | self.llm
            | self.modules['StrOutputParser']()
        ).invoke({"context": context, "question": improved_question})

        return final_answer

    def _handle_static_skills(self, question):
        """Handle questions that require static skills - provide all available info."""
        # STEP 1: Query Improvement - Improve the original question for better RAG efficiency
        original_question = question
        self.get_logger().info(f'Original question: {original_question}')
        improved_question = self.improve_query.invoke({"question": original_question})
        self.get_logger().info(f'Improved question: {improved_question}')

        context = f"""Información estática disponible:
    Hora actual: {self.current_time}
    Fecha actual: {self.current_date}"""

        static_prompt = self.modules['ChatPromptTemplate'].from_template(
            """Responde la pregunta del usuario usando la información proporcionada.

    {context}

    Pregunta: {question}
    """
        )

        static_chain = static_prompt | self.llm | self.modules['StrOutputParser']()
        return static_chain.invoke({"context": context, "question": improved_question})

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



        if task_result.dynamic_skill == "si" or task_result.dynamic_skill == "Si":
            self.get_logger().info('Routing to RAG chain')
            vector_result = self.vector_store_grader.invoke({"question": question})
            self.get_logger().info(f'Inmortal vector: {vector_result.inmortalVector}')
            self.get_logger().info(f'Reflex Vector: {vector_result.reflexVector}')

            if vector_result.inmortalVector == "si" or vector_result.inmortalVector == "Si":
                self.get_logger().info('Routing to Inmortal Vector Store')
                answer = self._create_rag_response(question, "vector1")
            elif vector_result.reflexVector == "si" or vector_result.reflexVector == "Si":
                self.get_logger().info('Routing to Reflex Vector Store')
                answer = self._create_rag_response(question, "vector2")
            else:
                self.get_logger().info('No specific vector store selected, defaulting to vector store 1')
                answer = self._create_rag_response(question, "vector1")

        elif task_result.static_skill == "si" or task_result.static_skill == "si":
            self.get_logger().info('Routing to static skills')
            answer = self._handle_static_skills(question)
        else:
            self.get_logger().info('Routing to basic LLM')
            original_question = question
            self.get_logger().info(f'Original question: {original_question}')
            improved_question = self.improve_query.invoke({"question": original_question})
            self.get_logger().info(f'Improved question: {improved_question}')

            basic_prompt = self.modules['ChatPromptTemplate'].from_template("Responde esta pregunta: {question}")
            basic_chain = basic_prompt | self.llm | self.modules['StrOutputParser']()
            answer = basic_chain.invoke({"question": improved_question})

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

