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

        Args:
            code (str): Python code string using 'db' to query Firestore.

        Returns:
            str: Query results or error message.
        """
        try:
            local_vars = {"db": self.db}
            exec(code, {}, local_vars)
            return local_vars.get("result", "No result returned")
        except Exception as e:
            return f"Error running Firestore query: {e}"

    def _initialize_agent(self):
        # Define tool as a plain function decorated with @tool
        @tool
        def firestore_tool(code: str) -> str:
            """
            Execute a Firestore query using Python code.

            Args:
                code (str): Python code string that uses the `db` object to query Firestore.
            
            Returns:
                str: Query results or an error message.
            """
            return self._firestore_query_runner(code)

        return CodeAgent(
            model=self.model,
            tools=[firestore_tool],
            additional_authorized_imports=["pandas", "numpy"],
            max_steps=10,
        )

    def ask(self, natural_language_query: str) -> str:
        """
        Ask the AI agent a natural language query.

        Args:
            natural_language_query (str): User's query in natural language.

        Returns:
            str: Agent's response in natural language.
        """
        return self.agent.run(natural_language_query)