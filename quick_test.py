"""
Quick test for your RAG system - just update the PDF path!
"""

from simple_rag_claude import SimpleRAGSystem

# Your PDF file path
PDF_PATH = "Hands On Machine Learning with Scikit Learn and TensorFlow (1).pdf"

def quick_test():
    print("🚀 Quick RAG Test")
    print(f"📄 Using PDF: {PDF_PATH}")
    
    # Initialize system
    rag = SimpleRAGSystem()
    
    try:
        # Upload your document
        print("\n📤 Uploading document...")
        doc_id = rag.function_1_pdf_to_pinecone(PDF_PATH, "My Test Document")
        print(f"✅ Success! Document ID: {doc_id[:8]}...")
        
        # Ask a question
        print("\n❓ Testing question...")
        answer = rag.generate_answer("What is this document about?")
        print(f"💡 Answer: {answer['answer']}")
        
        # List documents
        print("\n📋 Documents in system:")
        docs = rag.list_documents()
        for doc in docs:
            print(f"   📄 {doc['filename']} ({doc['chunk_count']} chunks)")
        
        print("\n🎉 Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure:")
        print("1. Update PDF_PATH with your actual file path")
        print("2. Check your .env file has correct API keys")
        print("3. Pinecone index exists with dimension 1024")

if __name__ == "__main__":
    quick_test()