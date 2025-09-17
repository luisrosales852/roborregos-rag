import os
print("1. Starting simple test")

# Test basic imports first
try:
    from langchain_openai import OpenAIEmbeddings
    print("2. OpenAI imports work")
except Exception as e:
    print(f"2. OpenAI import failed: {e}")

# Test if we can create embeddings without API key
try:
    print("3. Attempting to create embeddings...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    print("4. Embeddings object created (but may fail on actual API call)")
except Exception as e:
    print(f"4. Embeddings creation failed: {e}")

# Test PDF loading
try:
    from langchain_community.document_loaders import PyPDFLoader
    print("5. PDF loader imported")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf1_path = os.path.join(script_dir, "knowledge1.pdf")
    print(f"6. PDF path: {pdf1_path}")
    print(f"7. PDF exists: {os.path.exists(pdf1_path)}")
    
    if os.path.exists(pdf1_path):
        loader = PyPDFLoader(pdf1_path)
        docs = loader.load()
        print(f"8. PDF loaded successfully: {len(docs)} pages")
        print(f"9. First page preview: {docs[0].page_content[:100]}...")
    else:
        print("8. PDF file not found")
        
except Exception as e:
    print(f"5-9. PDF loading failed: {e}")
    import traceback
    traceback.print_exc()

print("10. Test completed")
