import tiktoken


class Indexing:
    """Text chunking strategies for document processing"""
    
    @staticmethod
    def chunk_by_tokens(text: str, max_tokens: int = 512, overlap_tokens: int = 50) -> List[str]:
        """Chunk text by token count with overlap"""

        
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            # Move start position with overlap
            start = end - overlap_tokens
            if start >= len(tokens) - overlap_tokens:
                break
        
        return chunks
    
    @staticmethod
    def chunk_by_sentences(text: str, max_chunk_size: int = 1000, overlap_size: int = 200) -> List[str]:
        """Chunk text by sentences with character limit"""
        import re
        
        # Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > max_chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                
                # Keep overlap
                overlap_chunk = []
                overlap_size_count = 0
                for s in reversed(current_chunk):
                    if overlap_size_count + len(s) <= overlap_size:
                        overlap_chunk.insert(0, s)
                        overlap_size_count += len(s)
                    else:
                        break
                
                current_chunk = overlap_chunk
                current_size = overlap_size_count
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

