from google.cloud import firestore
from google.oauth2 import service_account
import os
import firebase_admin


credentials = service_account.Credentials.from_service_account_file(
   os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
)
if not firebase_admin._apps:
    firebase_admin.initialize_app(credentials)
db = firestore.Client(database='default', credentials=credentials)

def get_collection_docs(collection_name):
    docs = db.collection(collection_name).stream()
    for doc in docs:
        print(f'\n{doc.id} => {doc.to_dict()}\n')

get_collection_docs('companies')
