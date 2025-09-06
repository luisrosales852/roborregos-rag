from typing import List, Dict, Tuple, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
import json
import math

class VectorStore:
    """PostgreSQL Vector Database handler using pgvector extension"""
    
    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        self.conn = None
        self.cursor = None
        self.connect()
        self.setup_vector_extension()
        self.create_tables()
    
    def set_search_parameters(self, probes: int = 10):
        """Configure IVFFlat search parameters for accuracy vs speed tradeoff"""
        try:
            # Set number of lists to search (default is 1, higher = more accurate but slower)
            self.cursor.execute(f"SET ivfflat.probes = {probes};")
            print(f"✓ Set IVFFlat probes to {probes}")
        except Exception as e:
            print(f"✗ Failed to set search parameters: {e}")
            raise

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
                    embedding vector(3072),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)            

            self.conn.commit()
            
            print("✓ Database tables created")
        except Exception as e:
            print(f"✗ Failed to create tables: {e}")
            raise
    
    def calculate_optimal_clusters(self, row_count: int) -> int:
        """Calculate optimal number of clusters based on sqrt(rows) rule"""
        if row_count < 10:
            # For small datasets, use brute force search (no index)
            return 0
        
        # Use sqrt rule but with reasonable bounds
        optimal = int(math.sqrt(row_count))
        
        # IVFFlat works best with clusters between 10 and 10000
        # Also ensure clusters don't exceed rows/10 to avoid empty clusters
        min_clusters = max(10, min(optimal, row_count // 10))
        max_clusters = min(10000, row_count // 2)
        
        return max(min_clusters, min(optimal, max_clusters))

    def manage_vector_index(self):
        """Create or recreate vector index based on current data size"""
        try:
            row_count = self.get_document_count()
            optimal_clusters = self.calculate_optimal_clusters(row_count)
            
            if optimal_clusters == 0:
                print(f"✓ Using brute force search for {row_count} documents (no index needed)")
                return
            
            # Check if index exists and get its current cluster count
            self.cursor.execute("""
                SELECT indexname, indexdef 
                FROM pg_indexes 
                WHERE tablename = 'documents' 
                AND indexname = 'documents_embedding_idx';
            """)
            
            existing_index = self.cursor.fetchone()
            needs_recreation = False
            
            if existing_index:
                print("Trying to use manage_vector_index")
                # Extract current cluster count from index definition
                index_def = existing_index['indexdef']
                if f"lists = {optimal_clusters}" not in index_def:
                    print(f"✓ Recreating index: current clusters suboptimal for {row_count} rows")
                    self.cursor.execute("DROP INDEX IF EXISTS documents_embedding_idx;")
                    needs_recreation = True
                else:
                    print(f"✓ Index already optimal with {optimal_clusters} clusters for {row_count} rows")
                    return
            else:
                needs_recreation = True
                print(f"✓ Creating new index with {optimal_clusters} clusters for {row_count} rows")
            
            if needs_recreation:
                # Create new index with optimal cluster count
                self.cursor.execute(f"""
                    CREATE INDEX documents_embedding_idx 
                    ON documents USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = {optimal_clusters});
                """)
                
                # Update statistics for query planner
                self.cursor.execute("ANALYZE documents;")
                self.conn.commit()
                print(f"✓ Vector index created/updated with {optimal_clusters} clusters")
                
        except Exception as e:
            self.conn.rollback()
            print(f"✗ Failed to manage vector index: {e}")
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
            # Check if we need to update the index after insertion
            # Only check periodically to avoid overhead
            if doc_id % 100 == 0:  # Check every 100 insertions
                self.manage_vector_index()
            return doc_id
        
        except Exception as e:
            self.conn.rollback()
            print(f"✗ Failed to insert document: {e}")
            raise
    
    #Might be inneficient or irrelevant
    def bulk_insert_documents(self, documents: List[Tuple[str, List[float], Dict]]):
        """Insert multiple documents efficiently and update index once"""
        try:
            for content, embedding, metadata in documents:
                embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                self.cursor.execute("""
                    INSERT INTO documents (content, embedding, metadata)
                    VALUES (%s, %s, %s);
                """, (content, embedding_str, json.dumps(metadata) if metadata else None))
            
            self.conn.commit()
            
            # Update index after bulk insertion
            self.manage_vector_index()
            
            print(f"✓ Bulk inserted {len(documents)} documents")
            
        except Exception as e:
            self.conn.rollback()
            print(f"✗ Failed to bulk insert documents: {e}")
            raise
    
    def search_similar(self, query_embedding: List[float], top_k: int = 5, explain: bool = False) -> List[Dict]:
        """Search for similar documents using cosine similarity"""
        try:
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            if explain:
                self.cursor.execute("""
                    EXPLAIN (ANALYZE, BUFFERS)
                    SELECT id, content, metadata,
                        1 - (embedding <=> %s::vector) AS similarity
                FROM documents
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, (embedding_str, embedding_str, top_k))
            
                plan = self.cursor.fetchall()
                print("\n📊 Query Execution Plan:")
                for row in plan:
                    print(row)
                print()
            
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
            self.cursor.execute("DROP INDEX IF EXISTS documents_embedding_idx;")
            self.conn.commit()
            print("✓ Database cleared")
        except Exception as e:
            self.conn.rollback()
            print(f"✗ Failed to clear database: {e}")
            raise
    
    def get_document_count(self) -> int:
        """Get the current number of documents in the database"""
        try:
            self.cursor.execute("SELECT COUNT(*) FROM documents;")
            return self.cursor.fetchone()['count']
        except Exception as e:
            print(f"✗ Failed to get document count: {e}")
            return 0

    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    