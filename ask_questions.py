"""
Simple Q&A script for your RAG system
Documents are already in Pinecone - just ask questions!
"""

from simple_rag_claude import SimpleRAGSystem

def ask_questions():
    print("🤖 RAG Q&A System")
    print("Your documents are already loaded. Ask any questions!")
    print("Type 'quit' or 'exit' to stop\n")
    
    try:
        # Initialize system (connects to existing Pinecone data)
        print("🔗 Connecting to your knowledge base...")
        rag = SimpleRAGSystem()
        
        # Show available documents
        print("\n📋 Available documents:")
        docs = rag.list_documents()
        for doc in docs:
            print(f"   📄 {doc['filename']} ({doc['chunk_count']} chunks)")
        
        print(f"\n✅ Ready! Found {len(docs)} document(s) with {sum(doc['chunk_count'] for doc in docs)} total chunks")
        print("=" * 60)
        
        # Q&A loop
        while True:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                print("Please enter a question!")
                continue
            
            print("\n🤔 Thinking...", end="", flush=True)
            
            try:
                # Get answer
                result = rag.generate_answer(question)
                
                print("\r" + " " * 15 + "\r", end="")  # Clear "Thinking..."
                
                print(f"💡 Answer:")
                print(f"{result['answer']}")
                
                # Show sources
                if result['sources']:
                    print(f"\n📚 Sources:")
                    for i, source in enumerate(result['sources'][:3], 1):  # Show top 3 sources
                        print(f"   {i}. {source['filename']} (Page {source['page_number']}) - Relevance: {source['relevance_score']:.2f}")
                
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again with a different question.")
    
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        print("\n💡 Make sure:")
        print("1. Your .env file has correct API keys")
        print("2. Documents are uploaded to Pinecone")
        print("3. AWS Bedrock access is configured")

if __name__ == "__main__":
    ask_questions()