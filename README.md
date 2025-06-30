# 🩺 FAST MEDICAL Chatbot

![Medical Bot](https://cdn.dribbble.com/users/29678/screenshots/2407580/media/34ee4b818fd4ddb3a616c91ccf4d9cfc.png)

![Result](https://github.com/Baldezo313/FastMedicalBot_RAG_Langchain/blob/main/Result.png)
## Get Answers in Seconds

This repository contains a medical chatbot powered by the Medical-Llama3-8B model. The chatbot is designed to assist users with medical queries using a conversational retrieval chain.

## Features

- Utilizes the `Medical-Llama3-8B` model for medical information.
- Incorporates document embeddings and a vector store for improved information retrieval.
- Provides a user-friendly interface with Gradio.
- Customizable model parameters through the Gradio interface.

## Setup

### Prerequisites

- Python 3.7 or higher
- CUDA-compatible GPU (for running the model efficiently)

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Baldezo313/FastMedicalBot_RAG_Langchain.git
    cd FastMedicalBot_RAG_Langchain
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Install the `bitsandbytes`, `transformers`, and `accelerate` packages:

    ```bash
    pip install --upgrade bitsandbytes transformers accelerate -i https://pypi.org/simple/
    ```

### Using the Model

`Biomistral'.

## Usage

1. You can Run the Jupyter cell code.

2. You can Open the provided link to access the Gradio interface and test the chatbot:  
https://huggingface.co/spaces/Baldezo313/medical_chatbot_space.

### Example Usage

Here's an example of how to use the chatbot:

- Type your question in the provided textbox.
- Adjust the parameters like `Max New Tokens`, `Temperature`, and `Context Length` as needed.
- Get the response from the chatbot displayed in a scrollable box.



## Acknowledgments

- The `Biomistral` 
- [Gradio](https://gradio.app/) for providing a user-friendly interface for machine learning models
