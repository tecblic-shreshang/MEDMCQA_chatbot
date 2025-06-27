import pandas as pd
import chromadb
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import warnings
import json
from typing import List, Dict, Optional

warnings.filterwarnings('ignore')
print("Libraries imported successfully!")

class SimpleMedMCQAProcessor:
    def __init__(self, max_samples=2000):
        """Initialize with reduced dataset size for faster processing"""
        self.max_samples = max_samples
        self.data = None
        
    def load_and_process_dataset(self):
        """Load and preprocess MedMCQA dataset"""
        print("Loading MedMCQA dataset...")
        
        try:
            # Load dataset
            dataset = load_dataset("openlifescienceai/medmcqa")
            
            # Use smaller subset for faster processing
            train_data = dataset['train'].to_pandas().head(self.max_samples)
            
            print(f"Loaded {len(train_data)} medical questions")
            
            # Preprocess data
            self.data = self._preprocess_data(train_data)
            return self.data
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            # Fallback to dummy data for testing
            return self._create_dummy_data()
    
    def _preprocess_data(self, df):
        """Preprocess the dataset"""
        processed_data = []
        
        for idx, row in df.iterrows():
            # Create comprehensive text for better retrieval
            question_text = f"Question: {row['question']}"
            options_text = f"Options: A) {row['opa']} B) {row['opb']} C) {row['opc']} D) {row['opd']}"
            
            # Get correct answer
            correct_option = row['cop']
            if pd.notna(correct_option):
                correct_answer = row[f"op{correct_option.lower()}"]
            else:
                correct_answer = "Answer not available"
            
            # Combine all text
            full_text = f"{question_text} {options_text}"
            
            processed_data.append({
                'id': str(idx),
                'question': row['question'],
                'full_text': full_text,
                'correct_answer': correct_answer,
                'subject': row.get('subject_name', 'General'),
                'topic': row.get('topic_name', 'General'),
                'options': {
                    'A': row['opa'],
                    'B': row['opb'], 
                    'C': row['opc'],
                    'D': row['opd']
                },
                'correct_option': correct_option
            })
        
        print(f"Processed {len(processed_data)} questions")
        return processed_data
    
    def _create_dummy_data(self):
        """Create dummy medical data if dataset loading fails"""
        print("Creating dummy medical data for testing...")
        
        dummy_data = [
            {
                'id': '1',
                'question': 'What is the normal range for blood pressure?',
                'full_text': 'Question: What is the normal range for blood pressure? Options: A) 120/80 mmHg B) 140/90 mmHg C) 160/100 mmHg D) 180/110 mmHg',
                'correct_answer': '120/80 mmHg',
                'subject': 'Medicine',
                'topic': 'Cardiology',
                'options': {'A': '120/80 mmHg', 'B': '140/90 mmHg', 'C': '160/100 mmHg', 'D': '180/110 mmHg'},
                'correct_option': 'A'
            },
            {
                'id': '2', 
                'question': 'Which vitamin deficiency causes scurvy?',
                'full_text': 'Question: Which vitamin deficiency causes scurvy? Options: A) Vitamin A B) Vitamin C C) Vitamin D D) Vitamin K',
                'correct_answer': 'Vitamin C',
                'subject': 'Medicine',
                'topic': 'Nutrition',
                'options': {'A': 'Vitamin A', 'B': 'Vitamin C', 'C': 'Vitamin D', 'D': 'Vitamin K'},
                'correct_option': 'B'
            }
        ]
        
        return dummy_data


class MedMCQAChromaStore:
    def __init__(self, collection_name="medmcqa", embedding_model="all-MiniLM-L6-v2"):
        """Initialize ChromaDB vector store"""
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB client
        self.client = chromadb.Client()
        self.collection = None
        
    def create_collection(self, data: List[Dict]):
        """Create ChromaDB collection and add documents"""
        print("Creating ChromaDB collection...")
        
        try:
            # Delete existing collection if it exists
            try:
                self.client.delete_collection(name=self.collection_name)
                print("Deleted existing collection")
            except:
                pass
            
            # Create new collection
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "MedMCQA medical questions and answers"}
            )
            
            print(f"Created collection: {self.collection_name}")
            
            # Prepare data for ChromaDB
            documents = []
            metadatas = []
            ids = []
            
            for item in data:
                documents.append(item['full_text'])
                metadatas.append({
                    'question': item['question'],
                    'correct_answer': item['correct_answer'],
                    'subject': item['subject'],
                    'topic': item['topic'],
                    'correct_option': str(item['correct_option']),
                    'options': json.dumps(item['options'])
                })
                ids.append(item['id'])
            
            # Add documents to collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"Added {len(documents)} documents to collection")
            return True
            
        except Exception as e:
            print(f"Error creating collection: {e}")
            return False
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for similar medical questions"""
        if self.collection is None:
            return []
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            # Format results
            formatted_results = []
            
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0], 
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    formatted_results.append({
                        'rank': i + 1,
                        'score': 1 - distance,  # Convert distance to similarity score
                        'question': metadata['question'],
                        'correct_answer': metadata['correct_answer'],
                        'subject': metadata['subject'],
                        'topic': metadata['topic'],
                        'options': json.loads(metadata['options']),
                        'correct_option': metadata['correct_option']
                    })
            
            return formatted_results
            
        except Exception as e:
            print(f"Search error: {e}")
            return []

class SimpleMedicalChatbot:
    def __init__(self, vector_store: MedMCQAChromaStore):
        self.vector_store = vector_store
        self.medical_keywords = [
            'disease', 'symptoms', 'treatment', 'medicine', 'diagnosis',
            'patient', 'medical', 'health', 'doctor', 'hospital', 'clinic',
            'surgery', 'medication', 'drug', 'therapy', 'infection', 'syndrome',
            'clinical', 'anatomy', 'physiology', 'pathology', 'cancer', 'tumor',
            'blood', 'heart', 'lung', 'kidney', 'liver', 'brain', 'bone',
            'diabetes', 'hypertension', 'fever', 'pain', 'headache', 'nausea',
            'vitamin', 'pressure', 'normal', 'range', 'deficiency'
        ]
    
    def is_medical_query(self, query: str) -> bool:
        """Check if the query is medical-related"""
        query_lower = query.lower()
        
        # Check for medical keywords
        has_medical_keywords = any(keyword in query_lower for keyword in self.medical_keywords)
        
        # Check for question patterns
        question_patterns = ['what is', 'what are', 'how to', 'causes of', 'symptoms of', 'treatment for', 'which']
        has_question_pattern = any(pattern in query_lower for pattern in question_patterns)
        
        return has_medical_keywords or has_question_pattern
    
    def chat(self, query: str) -> str:
        """Main chat function"""
        # Check if query is medical
        if not self.is_medical_query(query):
            return ("I can only answer medical questions based on the MedMCQA dataset. "
                   "Please ask a medical question about diseases, symptoms, treatments, etc.")
        
        # Search for relevant documents
        results = self.vector_store.search(query, top_k=3)
        
        if not results:
            return ("I couldn't find relevant information in the medical dataset. "
                   "Please try rephrasing your question or ask about a different medical topic.")
        
        # Generate response
        return self._generate_response(query, results)
    
    def _generate_response(self, query: str, results: List[Dict]) -> str:
        """Generate simplified response showing only the correct answers"""
        answers = [result['correct_answer'] for result in results[:1]]  # Top 2 answers
        unique_answers = list(dict.fromkeys(answers))  # Remove duplicates while preserving order
    
        response = "Here’s the medical answer based on our dataset:\n\n"
        for i, ans in enumerate(unique_answers, 1):
            response += f"✔️ Answer {i}: {ans}\n"
    
        response += "\n⚠️ *This information is from educational datasets and should not replace professional medical advice.*"
        return response


def initialize_chatbot():
    """Initialize the complete chatbot system"""
    print("🏥 Initializing MedMCQA Chatbot System...")
    
    # Step 1: Process data
    print("\n=== Step 1: Processing Data ===")
    processor = SimpleMedMCQAProcessor(max_samples=1000)  # Reduced for faster setup
    data = processor.load_and_process_dataset()
    
    if not data:
        print("❌ Failed to load data")
        return None
    
    # Step 2: Create vector store
    print("\n=== Step 2: Creating Vector Store ===")
    vector_store = MedMCQAChromaStore()
    
    if not vector_store.create_collection(data):
        print("❌ Failed to create vector store")
        return None
    
    # Step 3: Create chatbot
    print("\n=== Step 3: Creating Chatbot ===")
    chatbot = SimpleMedicalChatbot(vector_store)
    
    print("\n✅ MedMCQA Chatbot is ready!")
    return chatbot

def test_chatbot(chatbot):
    """Test the chatbot with sample queries"""
    if chatbot is None:
        print("❌ Chatbot not initialized")
        return
    
    print("\n🧪 Testing the MedMCQA Chatbot...\n")
    
    test_queries = [
        "What is the normal blood pressure?",
        "Which vitamin deficiency causes scurvy?",
        "What are symptoms of diabetes?",
        "How to treat hypertension?",
        "Tell me about heart disease",
        "What's the weather today?",  # Non-medical query
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"{'='*60}")
        print(f"Test {i}: {query}")
        print(f"{'='*60}")
        
        response = chatbot.chat(query)
        print(response)
        print()

def interactive_chat(chatbot):
    """Interactive chat function"""
    if chatbot is None:
        print("❌ Chatbot not initialized")
        return
    
    print("\n🏥 MedMCQA Chatbot - Interactive Mode")
    print("Ask medical questions or type 'quit' to exit\n")
    
    while True:
        try:
            query = input("You: ").strip()
            
            if query.lower() in ['quit', 'exit', 'bye', 'q']:
                print("👋 Goodbye!")
                break
                
            if not query:
                continue
                
            print(f"\n🤖 Chatbot:")
            response = chatbot.chat(query)
            print(response)
            print("\n" + "-"*60 + "\n")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    # Initialize the chatbot
    chatbot = initialize_chatbot()
    
    # Run tests
    # test_chatbot(chatbot)
    
    # Start interactive mode (uncomment to use)
    interactive_chat(chatbot)
    
    print("\n🎉 MedMCQA Chatbot setup completed!")
    print("The chatbot uses ChromaDB for vector storage and is grounded in medical data.")
    print("It will only answer medical questions and reject non-medical queries.")

