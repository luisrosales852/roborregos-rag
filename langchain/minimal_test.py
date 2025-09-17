#!/usr/bin/env python3
import os
import sys

print("Python version:", sys.version)
print("Current directory:", os.getcwd())

# Test basic functionality step by step
try:
    print("\n1. Testing basic imports...")
    from langchain_community.document_loaders import PyPDFLoader
    print("✓ PDF loader imported")
    
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    print("✓ Text splitter imported")
    
    from langchain_core.documents import Document
    print("✓ Document class imported")
    
    from langchain_chroma import Chroma
    print("✓ Chroma imported")
    
    print("\n2. Testing PDF loading...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(script_dir, "knowledge1.pdf")
    print(f"PDF path: {pdf_path}")
    print(f"PDF exists: {os.path.exists(pdf_path)}")
    
    if os.path.exists(pdf_path):
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        print(f"✓ PDF loaded: {len(docs)} pages")
        print(f"First page length: {len(docs[0].page_content)} characters")
    else:
        print("✗ PDF file not found")
        sys.exit(1)
    
    print("\n3. Testing text splitting...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    print(f"✓ Documents split into {len(splits)} chunks")
    print(f"First chunk length: {len(splits[0].page_content)} characters")
    
    print("\n4. Testing Chroma without embeddings...")
    # Try to create a Chroma instance without embeddings first
    try:
        # This might fail if we need embeddings
        vectorstore = Chroma(collection_name="test_collection", persist_directory="./test_chroma_db")
        print("✓ Chroma instance created")
        
        # Try to add documents (this will likely fail without embeddings)
        try:
            vectorstore.add_documents(splits[:2])  # Just try first 2 chunks
            print("✓ Documents added to Chroma")
        except Exception as e:
            print(f"✗ Failed to add documents: {e}")
            print("This is expected without proper embeddings")
            
    except Exception as e:
        print(f"✗ Chroma creation failed: {e}")
    
    print("\n5. Testing OpenAI embeddings...")
    try:
        from langchain_openai import OpenAIEmbeddings
        print("✓ OpenAI embeddings imported")
        
        # Check if API key is available
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            print("✓ OpenAI API key found")
            embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
            print("✓ Embeddings object created")
        else:
            print("✗ No OpenAI API key found in environment")
            print("You need to set OPENAI_API_KEY environment variable")
            
    except Exception as e:
        print(f"✗ OpenAI embeddings failed: {e}")
    
    print("\nTest completed successfully!")
    
except Exception as e:
    print(f"\n✗ Test failed with error: {e}")
    import traceback
    traceback.print_exc()
