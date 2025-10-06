"""
Test script for the Simple RAG System
Run this to test document upload and querying
"""

from simple_rag_claude import SimpleRAGSystem
import os

def main():
    print("🚀 Starting RAG System Test...")
    
    try:
        # Initialize the RAG system
        print("📡 Initializing RAG system...")
        rag = SimpleRAGSystem()
        print("✅ RAG system initialized successfully!")
        print(f"📊 Using models:")
        print(f"   - Embedding: {rag.embedding_model}")
        print(f"   - Chat: {rag.chat_model}")
        print(f"   - Pinecone Index: {rag.index_name}")
        
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        print("\n🔧 Make sure you have:")
        print("1. Valid AWS credentials (run 'aws configure' or set environment variables)")
        print("2. Pinecone API key in .env file")
        print("3. Pinecone index 'rag-faq' created with dimension 1024")
        print("4. AWS Bedrock access to Titan and Claude models")
        return
    
    # Look for PDF files in the current directory
    pdf_files = [f for f in os.listdir('.') if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("\n📄 No PDF files found in current directory.")
        print("Please add a PDF file to test with, or specify the full path below.")
        pdf_path = input("Enter path to your PDF file: ").strip()
        if not pdf_path or not os.path.exists(pdf_path):
            print("❌ Invalid file path. Exiting.")
            return
    else:
        print(f"\n📄 Found PDF files: {pdf_files}")
        if len(pdf_files) == 1:
            pdf_path = pdf_files[0]
            print(f"✅ Using: {pdf_path}")
        else:
            print("Which file would you like to use?")
            for i, file in enumerate(pdf_files, 1):
                print(f"{i}. {file}")
            try:
                choice = int(input("Enter number: ")) - 1
                pdf_path = pdf_files[choice]
                print(f"✅ Using: {pdf_path}")
            except (ValueError, IndexError):
                pdf_path = pdf_files[0]
                print(f"✅ Using first file: {pdf_path}")
    
    # Test document upload
    print(f"\n📤 Testing document upload...")
    try:
        doc_id = rag.function_1_pdf_to_pinecone(pdf_path)
        print(f"✅ Document uploaded successfully!")
        print(f"📋 Document ID: {doc_id}")
        
    except Exception as e:
        print(f"❌ Failed to upload document: {e}")
        return
    
    # List documents to verify upload
    print(f"\n📋 Checking uploaded documents...")
    try:
        docs = rag.list_documents()
        print(f"✅ Found {len(docs)} document(s) in database:")
        for doc in docs:
            print(f"   📄 {doc['filename']}")
            print(f"      ID: {doc['document_id'][:8]}...")
            print(f"      Chunks: {doc['chunk_count']}")
            print(f"      Created: {doc['created_at'][:19] if doc['created_at'] else 'Unknown'}")
            if doc.get('updated_at'):
                print(f"      Updated: {doc['updated_at'][:19]}")
    except Exception as e:
        print(f"⚠️ Could not list documents: {e}")
    
    # Test querying
    print(f"\n❓ Testing question answering...")
    
    # Sample questions - you can modify these based on your document
    sample_questions = [
        "What is this document about?",
        "What are the main topics covered?",
        "Can you summarize the key points?",
        "What information is provided?"
    ]
    
    print("Sample questions to try:")
    for i, q in enumerate(sample_questions, 1):
        print(f"{i}. {q}")
    
    print("\nEnter your question (or press Enter to use first sample question):")
    user_question = input("❓ Question: ").strip()
    
    if not user_question:
        user_question = sample_questions[0]
        print(f"Using sample question: {user_question}")
    
    try:
        print(f"\n🔍 Searching for relevant information...")
        answer_result = rag.generate_answer(user_question, top_k=3)
        
        print(f"\n✅ Answer generated successfully!")
        print(f"\n❓ Question: {answer_result['question']}")
        print(f"\n💬 Answer: {answer_result['answer']}")
        
        if answer_result['sources']:
            print(f"\n📚 Sources ({len(answer_result['sources'])} found):")
            for i, source in enumerate(answer_result['sources'], 1):
                print(f"   {i}. {source['filename']} (Page {source['page_number']})")
                print(f"      Relevance score: {source['relevance_score']:.3f}")
        else:
            print(f"\n📚 No relevant sources found")
            
    except Exception as e:
        print(f"❌ Failed to generate answer: {e}")
        return
    
    # Interactive Q&A
    print(f"\n🎯 Interactive Q&A mode (type 'quit' to exit):")
    while True:
        question = input("\n❓ Your question: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        if not question:
            continue
            
        try:
            result = rag.generate_answer(question, top_k=3)
            print(f"\n💬 Answer: {result['answer']}")
            
            if result['sources']:
                print(f"📚 Sources: {len(result['sources'])} relevant chunks found")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n🎉 Test completed successfully!")
    print(f"Your RAG system is working! 🚀")

if __name__ == "__main__":
    main()