import os
import torch
import gradio as gr

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.llm import LLMChain

# Load PDF
loader = PyPDFLoader("Medical_Book.pdf")
documents = loader.load()

# Split text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
all_splits = text_splitter.split_documents(documents)

# Embeddings
model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"})
vectorstores = FAISS.from_documents(all_splits, embeddings)

# Load LLM with quantization
llm_model = "ritvik77/Medical_Doctor_AI_LoRA-Mistral-7B-Instruct_FullModel"
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(llm_model, quantization_config=bnb_config, trust_remote_code=True, use_cache=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(llm_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=150, temperature=0.7, top_p=0.9, do_sample=True)
llm = HuggingFacePipeline(pipeline=pipe)

# Prompt templates
condense_prompt = PromptTemplate(
    input_variables=["question", "chat_history"],
    template="""
You are a helpful medical assistant. Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.

Chat History:
{chat_history}

Follow-up Question:
{question}

Standalone Question:"""
)
qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""
)
question_generator = LLMChain(llm=llm, prompt=condense_prompt)
combine_docs_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=qa_prompt)

def chatbot_response(user_input, max_new_tokens, temperature, context_length): 
    pipe.model.config.max_new_tokens = int(max_new_tokens)
    pipe.model.config.temperature = float(temperature)
    pipe.model.config.context_length = int(context_length)
    chain = ConversationalRetrievalChain(retriever=vectorstores.as_retriever(), combine_docs_chain=combine_docs_chain, question_generator=question_generator, return_source_documents=True)
    chat_history = []
    result = chain({"question": user_input, "chat_history": chat_history})
    return f"<div style='max-height: 400px; overflow-y: auto;'>{result['answer']}</div>"

interface = gr.Interface(
    fn=chatbot_response,
    inputs=[
        gr.Textbox(lines=2, placeholder="Type your question here...", label="Your Question", interactive=True),
        gr.Slider(label="Max New Tokens", minimum=1, maximum=2000, value=150, step=1, interactive=True),
        gr.Slider(label="Temperature", minimum=0.1, maximum=1.0, value=0.7, step=0.01, interactive=True),
        gr.Slider(label="Context Length", minimum=100, maximum=4000, value=2000, step=1, interactive=True)
    ],
    outputs=gr.HTML(label="Chatbot Response"),
    title="🩺 MEDICAL Chatbot",
    description="""
    <div style='text-align: center;'>
        <img src='https://cdn.dribbble.com/users/29678/screenshots/2407580/media/34ee4b818fd4ddb3a616c91ccf4d9cfc.png' alt='Medical Bot' width='100'>
        <p>Check the responses from the Medical Llama 3-8B model!</p>
    </div>
    """
)

interface.launch()
