import os
import tempfile
import logging
from typing import List, Tuple, Optional
from dotenv import load_dotenv

# LangChain imports
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.globals import set_llm_cache
from langchain_community.cache import RedisCache
import redis


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocBot:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        try:
            # Initialize Redis cache
            self.redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
            set_llm_cache(RedisCache(redis_=self.redis_client))
            logger.info(f"Redis cache initialized with host: localhost, port: 6379")

            self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0,
                convert_system_message_to_human=True
            )
            self.vectorstore: Optional[FAISS] = None
            self.qa_chain: Optional[RetrievalQA] = None
            self.document_chunks: List[Document] = []

            logger.info(f"DocumentChatbot initialized with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize chatbot: {str(e)}")
            raise

    def get_file_loader(self, file_path: str, file_type: str):
        if file_type == "application/pdf" or file_path.endswith('.pdf'):
            return PyPDFLoader(file_path)
        elif file_type in ["text/plain", "text/markdown"] or file_path.endswith(('.txt', '.md')):
            return TextLoader(file_path, encoding='utf-8')
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def text_splitter(self, documents: List[Document]) -> RecursiveCharacterTextSplitter:
        total_len = sum(len(doc.page_content) for doc in documents)
        if total_len > 100000:  # Very large documents
            chunk_size, overlap = 800, 150
            logger.info("Using small chunks for very large document")
        elif total_len > 50000:  # Large documents
            chunk_size, overlap = 1000, 200
            logger.info("Using medium chunks for large document")
        else:  # Regular documents
            chunk_size, overlap = 1200, 250
            logger.info("Using large chunks for regular document")

        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def create_custom_prompt(self) -> PromptTemplate:
        template = """You are a helpful document reader AI assistant that answers questions about the attached document based on the provided context.
        Use the following pieces of context to answer the question at the end.

        People are going to come to you with a a document which they dont understand. Your job is to read the document, understand it and explain it in a simple manner to the user. The document might be a complex one (eg. Law, Tech, Math) or a simple one,
        but your job is to provide them with every detail of that document in a way which is very very easy to understand for the user.
        
        Instructions:
        - If you don't know the answer based on the context, say "I don't have enough information to answer that question."
        - Provide specific details and examples from the context when possible
        - Be concise but comprehensive
        - If the question asks for a list, format it clearly
        - Always base your answer on the provided context
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:"""
        return PromptTemplate(template=template, input_variables=["context", "question"]) 
    
    def doc_loader(self, file_content: bytes, filename: str, file_type: str) -> dict:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp_file:
                tmp_file.write(file_content)
                tmp_path = tmp_file.name

            loader = self.get_file_loader(tmp_path, file_type)
            docs = loader.load()

            if not docs:
                raise ValueError("No content could be extracted from the document")
            
            txt_splitter = self.text_splitter(docs)
            self.document_chunks = txt_splitter.split_documents(docs)
            
            # Create vector store with batch processing for large documents
            self.vectorstore = self.create_vectorstore(self.document_chunks)

            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}
                ),
                return_source_documents=True,
                chain_type_kwargs={"prompt": self.create_custom_prompt()}
            )

            os.unlink(tmp_path)
            
            logger.info(f"Document processed: {len(self.document_chunks)} chunks created")
            
            return {
                "success": True,
                "chunk_count": len(self.document_chunks),
                "filename": filename,
                "total_characters": sum(len(chunk.page_content) for chunk in self.document_chunks)
            }
            
        except Exception as e:
            logger.error(f"Error loading document: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "chunk_count": 0
            }
        
    def create_vectorstore(self, chunks: List[Document], batch_size: int = 50) -> FAISS:
        if len(chunks) <= batch_size:
            return FAISS.from_documents(chunks, self.embeddings)
        
        logger.info(f"Processing {len(chunks)} chunks in batches of {batch_size}")
        
        # Create initial vectorstore with first batch
        vectorstore = FAISS.from_documents(chunks[:batch_size], self.embeddings)
        for i in range(batch_size, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            batch_vectorstore = FAISS.from_documents(batch, self.embeddings)
            vectorstore.merge_from(batch_vectorstore)
            logger.info(f"Processed batch {i//batch_size + 1}")
        
        return vectorstore
    
    def ask_question(self, question: str) -> dict:
        if not self.qa_chain:
            return {
                "success": False,
                "answer": "Please upload a document first!",
                "sources": []
            }
        
        if not question.strip():
            return {
                "success": False,
                "answer": "Please provide a valid question.",
                "sources": []
            }
        
        try:
            # Get answer from QA chain
            result = self.qa_chain({"query": question.strip()})
            
            # Extract source information
            sources = []
            if "source_documents" in result and result["source_documents"]:
                for i, doc in enumerate(result["source_documents"]):
                    sources.append({
                        "index": i + 1,
                        "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                        "metadata": doc.metadata
                    })
            
            return {
                "success": True,
                "answer": result["result"],
                "sources": sources,
                "question": question
            }
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return {
                "success": False,
                "answer": f"Error processing question: {str(e)}",
                "sources": []
            }
    def get_document_summary(self) -> dict:
        """Get summary information about the loaded document"""
        if not self.document_chunks:
            return {"loaded": False}
        
        total_chars = sum(len(chunk.page_content) for chunk in self.document_chunks)
        
        return {
            "loaded": True,
            "chunk_count": len(self.document_chunks),
            "total_characters": total_chars,
            "estimated_tokens": total_chars // 4,  # Rough estimate
            "vectorstore_ready": self.vectorstore is not None,
            "qa_chain_ready": self.qa_chain is not None
        }
    
    def clear_document(self):
        """Clear the currently loaded document"""
        self.vectorstore = None
        self.qa_chain = None
        self.document_chunks = []
        logger.info("Document cleared from memory")    

if __name__ == "__main__":
    # Test the chatbot
    chatbot = DocBot()
    
    # Test with a simple text file
    test_content = """
    This is a test document about artificial intelligence.
    AI is a rapidly growing field that includes machine learning,
    natural language processing, and computer vision.
    Machine learning uses algorithms to learn from data.
    """
    
    result = chatbot.load_document(
        test_content.encode('utf-8'),
        "test.txt",
        "text/plain"
    )
    
    if result["success"]:
        print(f"Document loaded successfully: {result['chunk_count']} chunks")
        
        # Test question
        response = chatbot.ask_question("What is AI?")
        if response["success"]:
            print(f"Answer: {response['answer']}")
        else:
            print(f"Error: {response['answer']}")
    else:
        print(f"Failed to load document: {result.get('error', 'Unknown error')}")
    