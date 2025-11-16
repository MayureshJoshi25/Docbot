# DocBot

DocBot is a document question-answering chatbot built upon LangChain, that allows users to upload PDF documents and ask questions about their content. It leverages the power of Large Language Models (LLMs) to provide accurate and relevant answers based on the context of the uploaded documents.

## Features

*   **PDF Upload:** Easily upload PDF documents for analysis.
*   **Question Answering:** Ask natural language questions about the document content.
*   **LLM Powered:** Utilizes advanced LLMs for intelligent responses.
*   **Contextual Understanding:** Provides answers grounded in the uploaded document's text.

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

*   Docker (and Docker Compose)
*   Python 3.9+

### Installation and Setup

1.  **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd docbot
    ```

2.  **Build and run with Docker Compose:**

    ```bash
    docker-compose up --build
    ```

    This will build the Docker images for the application and its dependencies, and then start the services. The application will be accessible at `http://localhost:8501`.

3.  **Manual Installation (Alternative):**

    If you prefer to run without Docker, you can set up the Python environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

    Then, you can run the Streamlit application:

    ```bash
    streamlit run app.py
    ```

## Usage

1.  **Access the application:** Open your web browser and navigate to `http://localhost:8501`.
2.  **Upload a PDF:** Use the sidebar to upload your PDF document.
3.  **Ask Questions:** Once the document is processed, you can type your questions into the chat interface and receive answers based on the document's content.

## Project Structure

*   `app.py`: The main Streamlit application script.
*   `docbot.py`: Contains the core logic for document processing and question answering.
*   `requirements.txt`: Lists Python dependencies.
*   `Dockerfile`: Defines the Docker image for the application.
*   `docker-compose.yml`: Defines how the application services are run using Docker Compose.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
