import os
import firebase_admin
from google.cloud import firestore
from google.oauth2 import service_account
from smolagents import CodeAgent, HfApiModel, tool
from huggingface_hub import login
from dotenv import load_dotenv, find_dotenv


class FirestoreAgentService:
    def __init__(self):
        self._load_environment()
        self.db = self._initialize_firestore()
        self.model = self._initialize_llm()
        self.collections_info = self._discover_collections()
        self.agent = self._initialize_agent()

    def _load_environment(self):
        load_dotenv(find_dotenv())
        huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not huggingface_api_key:
            raise EnvironmentError("HUGGINGFACE_API_KEY not set in environment")
        login(huggingface_api_key)

    def _initialize_firestore(self):
        credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if not credentials_path:
            raise EnvironmentError("GOOGLE_APPLICATION_CREDENTIALS not set in environment")

        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(credentials)
        return firestore.Client(database='default', credentials=credentials)

    def _discover_collections(self):
        """
        Discover all collections and sample their structure to understand the data schema.
        """
        collections_info = {}
        try:
            collections = self.db.collections()
            for collection_ref in collections:
                collection_name = collection_ref.id
                
                # Get a few sample documents to understand the structure
                sample_docs = list(collection_ref.limit(3).stream())
                
                if sample_docs:
                    # Analyze field types and structure
                    all_fields = set()
                    field_types = {}
                    sample_data = []
                    
                    for doc in sample_docs:
                        doc_data = doc.to_dict()
                        sample_data.append({k: str(v)[:50] + "..." if len(str(v)) > 50 else str(v) 
                                          for k, v in doc_data.items()})
                        
                        for field, value in doc_data.items():
                            all_fields.add(field)
                            field_types[field] = type(value).__name__
                    
                    collections_info[collection_name] = {
                        "fields": list(all_fields),
                        "field_types": field_types,
                        "sample_data": sample_data,
                        "doc_count_estimate": len(sample_docs)  # This is just from sample
                    }
                else:
                    collections_info[collection_name] = {
                        "fields": [],
                        "field_types": {},
                        "sample_data": [],
                        "doc_count_estimate": 0
                    }
                    
        except Exception as e:
            print(f"Error discovering collections: {e}")
            collections_info = {}
        print(f"Discovered collections: {collections_info.keys()}")
        return collections_info

    def _initialize_llm(self):
        return HfApiModel(
            model="Qwen/Qwen2.5-72B-Instruct",
            provider="together",
            max_tokens=4096,
            temperature=0.1,
        )

    def _firestore_query_runner(self, code: str) -> str:
        """
        Executes dynamically generated Firestore query code.
        """
        try:
            local_vars = {"db": self.db}
            exec(code, {}, local_vars)
            result = local_vars.get("result", "No result returned")
            
            # Convert result to string if it's a list or dict for better readability
            if isinstance(result, (list, dict)):
                return str(result)
            return result
        except Exception as e:
            return f"Error running Firestore query: {e}"

    def _initialize_agent(self):
        @tool
        def get_collections_info() -> str:
            """
            Get information about all available Firestore collections, their fields, and sample data.
            
            Returns:
                str: Detailed information about collections structure.
            """
            if not self.collections_info:
                return "No collections found or accessible."
            
            info_str = "Available Firestore Collections:\n\n"
            for collection_name, info in self.collections_info.items():
                info_str += f"Collection: {collection_name}\n"
                info_str += f"Fields: {', '.join(info['fields'])}\n"
                info_str += f"Field Types: {info['field_types']}\n"
                if info['sample_data']:
                    info_str += f"Sample Data: {info['sample_data'][0]}\n"
                info_str += f"Estimated Documents: {info['doc_count_estimate']}\n\n"
            
            return info_str

        @tool
        def find_best_collection(query_description: str) -> str:
            """
            Find the best collection(s) that would contain data relevant to answer the user's query.
            
            Args:
                query_description (str): Description of what the user is looking for.
            
            Returns:
                str: Recommended collection name(s) and reasoning.
            """
            if not self.collections_info:
                return "No collections available to analyze."
            
            recommendations = []
            for collection_name, info in self.collections_info.items():
                # Simple keyword matching - you can make this more sophisticated
                relevance_score = 0
                query_lower = query_description.lower()
                
                # Check if collection name is relevant
                if any(word in collection_name.lower() for word in query_lower.split()):
                    relevance_score += 3
                
                # Check if field names are relevant
                for field in info['fields']:
                    if any(word in field.lower() for word in query_lower.split()):
                        relevance_score += 2
                
                # Check sample data for relevance
                for sample in info['sample_data']:
                    sample_text = str(sample).lower()
                    if any(word in sample_text for word in query_lower.split()):
                        relevance_score += 1
                
                if relevance_score > 0:
                    recommendations.append({
                        'collection': collection_name,
                        'score': relevance_score,
                        'fields': info['fields']
                    })
            
            # Sort by relevance score
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            
            if recommendations:
                result = f"Best matching collections for '{query_description}':\n\n"
                for i, rec in enumerate(recommendations[:3], 1):  # Top 3 recommendations
                    result += f"{i}. {rec['collection']} (score: {rec['score']})\n"
                    result += f"   Available fields: {', '.join(rec['fields'])}\n\n"
                return result
            else:
                return f"No collections seem directly relevant to '{query_description}'. Available collections: {list(self.collections_info.keys())}"

        @tool
        def execute_firestore_query(collection_name: str, query_code: str) -> str:
            """
            Execute a Firestore query on a specific collection.
            
            Args:
                collection_name (str): Name of the collection to query.
                query_code (str): Python code to execute the query using db.collection(collection_name).
            
            Returns:
                str: Query results or error message.
            """
            try:
                # Validate collection exists
                if collection_name not in self.collections_info:
                    return f"Collection '{collection_name}' not found. Available: {list(self.collections_info.keys())}"
                
                # Execute the query code
                local_vars = {
                    "db": self.db,
                    "collection_name": collection_name,
                    "collection_ref": self.db.collection(collection_name)
                }
                exec(query_code, {}, local_vars)
                result = local_vars.get("result", "No result returned")
                
                # Format result for better readability
                if isinstance(result, list):
                    if len(result) > 10:  # Limit output size
                        formatted_result = f"Found {len(result)} results. Showing first 10:\n"
                        formatted_result += str(result[:10])
                    else:
                        formatted_result = f"Found {len(result)} results:\n{result}"
                else:
                    formatted_result = str(result)
                
                return formatted_result
            except Exception as e:
                return f"Error executing query on collection '{collection_name}': {e}"

        return CodeAgent(
            model=self.model,
            tools=[get_collections_info, find_best_collection, execute_firestore_query],
            additional_authorized_imports=["pandas", "numpy"],
            max_steps=15,
        )

    def ask(self, natural_language_query: str) -> str:
        """
        Ask the AI agent a natural language query using the improved workflow.
        """
        # Enhanced prompt to guide the agent through the improved workflow
        enhanced_query = f"""
        User Query: {natural_language_query}
        
        Please follow this workflow to answer the user's query:
        
        1. First, use get_collections_info() to understand what collections are available
        2. Use find_best_collection() to identify which collection(s) would best answer the user's query
        3. Use execute_firestore_query() with the recommended collection to run the appropriate Firestore query
        4. Provide a clear, natural language response based on the query results
        
        Make sure to explain your reasoning for choosing specific collections and provide context for the results.
        """
        
        return self.agent.run(enhanced_query)