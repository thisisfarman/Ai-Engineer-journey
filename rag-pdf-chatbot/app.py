from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from ibm_watsonx_ai import Credentials
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
import gradio as gr
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')
def get_llm():
    model_id='mistralai/mistral-medium-2505'
    params={
        GenParams.MAX_NEW_TOKENS:256,
        GenParams.TEMPERATURE:0.5
    }
    project_id="skills-network"
    watsonx_llm=WatsonxLLM(
        model_id=model_id,
        params=params,
        url="https://us-south.ml.cloud.ibm.com",
        project_id=project_id
    )
    return watsonx_llm

def document_loader(file):
    loader=PyPDFLoader(file.name)
    loaded_doc=loader.load()
    return loaded_doc
 
def text_splitter(data):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chuncks=text_splitter.split_documents(data)
    return chuncks

def vector_database(chunks):
    embedding_model=watsonx_embedding()
    vectordb=Chroma.from_documents(chunks,embedding_model)
    return vectordb

def watsonx_embedding():
    embed_params={
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS:3,
        EmbedTextParamsMetaNames.RETURN_OPTIONS:{"input_text":True}
    }
    watsonx_embedding = WatsonxEmbeddings(
        model_id="ibm/granite-embedding-278m-multilingual",
        url="https://us-south.ml.cloud.ibm.com",
        project_id="skills-network",
        params=embed_params,
    )
    return watsonx_embedding

def retriver(file):
    load=document_loader(file)
    chunks=text_splitter(load)
    vectordb=vector_database(chunks)
    retriver=vectordb.as_retriever()
    return retriver

def retriever_qa(file,query):
    llm=get_llm()
    retriver_obj=retriver(file)
    qa=RetrievalQA.from_chain_type(
        chain_type="stuff",
        retriever_obj=retriver_obj,
        return_source_documents=False
    )
    response=qa.invoke(query)
    return response['result']

rag_application = gr.Interface(
    fn=retriever_qa,
    allow_flagging="never",
    inputs=[
        gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath"),
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
    ],
    outputs=gr.Textbox(label="output"),
    title="RAG Chatbot",
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document."
)
rag_application.launch(server_name="0.0.0.0",server_port=7860,share=True)



