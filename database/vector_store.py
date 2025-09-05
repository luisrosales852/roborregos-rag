from typing import List, Dict, Tuple, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
import json


class VectorStore:
    """PostgreSQL Vector Database handler using pgvector extension"""
    
    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        self.conn = None
        self.cursor = None
        self.connect()
        self.setup_vector_extension()
        self.create_tables()
    
    def connect(self):
        """Establish connection to PostgreSQL database"""
        try:
            self.conn = psycopg2.connect(
                host=self.db_config.get('host', 'localhost'),
                port=int(self.db_config.get('port', '5432')),
                database=self.db_config.get('database', 'rag_db'),
                user=self.db_config.get('user', 'postgres'),
                password=self.db_config.get('password', 'password')
            )
            self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            print("✓ Connected to PostgreSQL database")
        except Exception as e:
            print(f"✗ Failed to connect to database: {e}")
            raise
    
    def setup_vector_extension(self):
        """Install pgvector extension if not already installed"""
        try:
            self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            self.conn.commit()
            print("✓ pgvector extension ready")
        except Exception as e:
            print(f"✗ Failed to setup pgvector: {e}")
            raise

    def create_tables(self):
        """Create necessary tables for storing documents and embeddings"""
        try:
            # Create documents table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata JSONB,
                    embedding vector(1536),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create index for vector similarity search
            self.cursor.execute("""
                CREATE INDEX IF NOT EXISTS documents_embedding_idx 
                ON documents USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
            
            self.conn.commit()
            print("✓ Database tables created")
        except Exception as e:
            print(f"✗ Failed to create tables: {e}")
            raise
    
    def insert_document(self, content: str, embedding: List[float], metadata: Dict = None):
        """Insert a document with its embedding into the database"""
        try:
            embedding_str = '[' + ','.join(map(str, embedding)) + ']'
            self.cursor.execute("""
                INSERT INTO documents (content, embedding, metadata)
                VALUES (%s, %s, %s)
                RETURNING id;
            """, (content, embedding_str, json.dumps(metadata) if metadata else None))
            
            doc_id = self.cursor.fetchone()['id']
            self.conn.commit()
            return doc_id
        except Exception as e:
            self.conn.rollback()
            print(f"✗ Failed to insert document: {e}")
            raise
    
    def search_similar(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """Search for similar documents using cosine similarity"""
        try:
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            self.cursor.execute("""
                SELECT id, content, metadata,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM documents
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, (embedding_str, embedding_str, top_k))
            
            results = self.cursor.fetchall()
            return results
        except Exception as e:
            print(f"✗ Failed to search documents: {e}")
            raise
    
    def clear_database(self):
        """Clear all documents from the database"""
        try:
            self.cursor.execute("TRUNCATE TABLE documents;")
            self.conn.commit()
            print("✓ Database cleared")
        except Exception as e:
            self.conn.rollback()
            print(f"✗ Failed to clear database: {e}")
            raise
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

