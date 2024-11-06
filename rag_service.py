import logging
from typing import List, Dict, Union
import numpy as np
import json
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class RAGService:
    def __init__(self, embeddings_dir: str):
        """Initialize RAG service to use existing embeddings
        
        Args:
            embeddings_dir: Directory containing the FAISS index and passages
        """
        self.embeddings_dir = embeddings_dir
        self.vectorstore = None
        self.embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.passages = []
        
    def load_index(self) -> bool:
        """Load existing FAISS index and passages
        
        Returns:
            bool: True if loading was successful, False otherwise
        """
        try:
            # Load the vector store
            self.vectorstore = FAISS.load_local(
                self.embeddings_dir,
                self.embedding_model,
                allow_dangerous_deserialization=True
            )
            
            # Load raw passages
            passages_path = os.path.join(self.embeddings_dir, 'passages.json')
            with open(passages_path, 'r') as f:
                self.passages = json.load(f)
                
            logging.info(f"Loaded existing index from {self.embeddings_dir}")
            return True
            
        except Exception as e:
            logging.error(f"Error loading index or passages: {e}")
            return False
        
    def get_relevant_context(self, query: str, k: int = 3) -> str:
        """Retrieve relevant context for a given query
        
        Args:
            query: The query text
            k: Number of relevant passages to retrieve
            
        Returns:
            str: Combined relevant passages as context
            
        Raises:
            ValueError: If index hasn't been loaded
        """
        if self.vectorstore is None:
            raise ValueError("Index not loaded. Call load_index() first.")
            
        # Get relevant documents
        docs = self.vectorstore.similarity_search(query, k=k)
        
        # Extract and combine the content
        context = "\n\n".join([doc.page_content for doc in docs])
        
        return context

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    service = RAGService('embeddings')
    
    if service.load_index():
        query = "How to install Python packages?"
        context = service.get_relevant_context(query)
        print(context)
    else:
        logging.error("Failed to load index")