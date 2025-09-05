def main():
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'rag_db',
        'user': 'postgres',
        'password': 'your_password'  # Change this
    }
    
    # Initialize RAG pipeline
    # You'll need to set OPENAI_API_KEY environment variable or pass it directly
    rag = RAGPipeline(db_config, openai_api_key=None)  # Will use env variable
    
    # Example: Ingest a knowledge base file
    # rag.ingest_knowledge_base("path/to/your/knowledge_base.txt", chunking_method="tokens")
    
    # Run interactive session
    # rag.interactive_session()
    
    print("RAG Pipeline initialized. Use the following methods:")
    print("- rag.ingest_knowledge_base('file_path.txt') to add documents")
    print("- rag.interactive_session() to start Q&A")
    print("- rag.query('your question') for programmatic access")


if __name__ == "__main__":
    main()