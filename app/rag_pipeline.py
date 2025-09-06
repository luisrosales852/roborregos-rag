

class RAGPipeline:
    """Main RAG Pipeline orchestrator"""
    
    def __init__(self, db_config: Dict[str, str], openai_api_key: Optional[str] = None):
        self.vector_db = PostgresVectorDB(db_config)
        self.embedding_model = EmbeddingModel(api_key=openai_api_key)
        self.client = OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
        self.chunk_strategy = ChunkingStrategy()
        self.chat_model = "gpt-3.5-turbo"  # Can be changed to gpt-4
    
    def ingest_knowledge_base(self, file_path: str, chunking_method: str = "tokens"):
        """Process and store knowledge base in vector database"""
        print(f"\n📚 Ingesting knowledge base from: {file_path}")
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Chunk the content
            if chunking_method == "tokens":
                chunks = self.chunk_strategy.chunk_by_tokens(content)
            else:
                chunks = self.chunk_strategy.chunk_by_sentences(content)
            
            print(f"✓ Created {len(chunks)} chunks")
            
            # Generate embeddings for all chunks
            print("🔄 Generating embeddings...")
            embeddings = self.embedding_model.embed_batch(chunks)
            
            # Store in database
            print("💾 Storing in vector database...")
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                metadata = {
                    "source": file_path,
                    "chunk_index": i,
                    "chunk_method": chunking_method
                }
                self.vector_db.insert_document(chunk, embedding, metadata)
            
            print(f"✓ Successfully ingested {len(chunks)} chunks into database")
            
        except Exception as e:
            print(f"✗ Failed to ingest knowledge base: {e}")
            raise
    
    def retrieve_context(self, query: str, top_k: int = 5, probes: int = 10) -> List[Dict]:
        self.vector_db.set_search_parameters(probes=probes)
        """Retrieve relevant context for a query"""
        # Generate query embedding
        query_embedding = self.embedding_model.embed_text(query)
        
        # Search similar documents
        results = self.vector_db.search_similar(query_embedding, top_k)
        
        return results
    
    def generate_answer(self, query: str, context: List[Dict]) -> str:
        """Generate answer using retrieved context"""
        # Prepare context string
        context_text = "\n\n".join([
            f"[Document {i+1}]:\n{doc['content']}"
            for i, doc in enumerate(context)
        ])
        
        # System prompt for RAG
        system_prompt = """You are a helpful AI assistant that answers questions based on the provided context.
        
Instructions:
- Answer the question using ONLY the information provided in the context
- If the context doesn't contain enough information to answer the question, say so
- Be concise and direct in your answers
- Quote relevant parts from the context when appropriate
- Do not make up information not present in the context"""
        
        # User prompt
        user_prompt = f"""Context:
{context_text}

Question: {query}

Answer:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            print(f"✗ Failed to generate answer: {e}")
            raise
    
    def query(self, question: str, top_k: int = 5) -> Tuple[str, List[Dict]]:
        """Main query interface - retrieve context and generate answer"""
        print(f"\n🔍 Processing query: {question}")
        
        # Retrieve relevant context
        context = self.retrieve_context(question, top_k)
        
        if not context:
            return "No relevant information found in the knowledge base.", []
        
        print(f"✓ Retrieved {len(context)} relevant chunks")
        
        # Generate answer
        answer = self.generate_answer(question, context)
        
        return answer, context
    
    def interactive_session(self):
        """Run an interactive Q&A session via terminal"""
        print("\n🤖 RAG Interactive Session Started")
        print("Type 'quit' or 'exit' to end the session\n")
        
        while True:
            try:
                # Get user question
                question = input("\n❓ Your question: ").strip()
                
                if question.lower() in ['quit', 'exit']:
                    print("\n👋 Goodbye!")
                    break
                
                if not question:
                    continue
                
                # Process query
                answer, context = self.query(question)
                
                # Display answer
                print(f"\n💡 Answer: {answer}")
                
                # Optionally show retrieved chunks
                show_context = input("\n📋 Show retrieved context? (y/n): ").strip().lower()
                if show_context == 'y':
                    print("\n📄 Retrieved Context:")
                    for i, doc in enumerate(context):
                        print(f"\n--- Chunk {i+1} (Similarity: {doc['similarity']:.3f}) ---")
                        print(doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content'])
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")