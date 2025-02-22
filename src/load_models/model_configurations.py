from openai import AzureOpenAI
import os
from dotenv import load_dotenv
from langchain_openai.chat_models.azure import AzureChatOpenAI
from langchain_openai.embeddings.azure import AzureOpenAIEmbeddings

load_dotenv("env.dev")

primary_model = os.getenv("PRIMARY_MODEL_NAME")
primary_model_deployment_name = os.getenv("PRIMARY_MODEL_DEPLOYMENT_NAME")
primary_model_openai_api_version = os.getenv("PRIMARY_MODEL_OPENAI_API_VERSION")
primary_model_azure_endpoint = os.getenv("PRIMARY_MODEL_AZURE_OPENAI_ENDPOINT")
primary_model_azure_openai_api_key = os.getenv("PRIMARY_MODEL_AZURE_OPENAI_API_KEY")

secondary_model = os.getenv("SECONDARY_MODEL_NAME")
secondary_model_deployment_name = os.getenv("SECONDARY_MODEL_DEPLOYMENT_NAME")
secondary_model_openai_api_version = os.getenv("SECONDARY_MODEL_OPENAI_API_VERSION")
secondary_model_azure_endpoint = os.getenv("SECONDARY_MODEL_AZURE_OPENAI_ENDPOINT")
secondary_model_azure_openai_api_key = os.getenv("SECONDARY_MODEL_AZURE_OPENAI_API_KEY")

embedding_model = os.getenv("EMBEDDING_MODEL")
embedding_model_deployment_name = os.getenv("EMBEDDING_MODEL_DEPLOYMENT_NAME")
embedding_model_openai_api_version = os.getenv("EMBEDDING_MODEL_OPENAI_API_VERSION")
embedding_model_azure_endpoint = os.getenv("EMBEDDING_MODEL_AZURE_OPENAI_ENDPOINT")
embedding_model_azure_openai_api_key = os.getenv("EMBEDDING_MODEL_AZURE_OPENAI_API_KEY")

def load_azureopenai_client():
    client = AzureOpenAI(
    api_key=primary_model_azure_openai_api_key,  
    api_version=primary_model_openai_api_version,
    base_url=f"{primary_model_azure_endpoint}openai/deployments/{primary_model_deployment_name}",)

    return client

def load_secondary_llm(temperature=0):
    llm = AzureChatOpenAI(
        openai_api_version=secondary_model_openai_api_version,
        azure_endpoint=secondary_model_azure_endpoint,
        azure_deployment=secondary_model_deployment_name,
        temperature=temperature,
        model=secondary_model,
        api_key=secondary_model_azure_openai_api_key)
    
    return llm

def load_primary_llm():
    llm = AzureChatOpenAI(
    openai_api_version=primary_model_openai_api_version,
    azure_endpoint=primary_model_azure_endpoint,
    azure_deployment=primary_model_deployment_name,
    temperature=0,
    model=primary_model,
    api_key=primary_model_azure_openai_api_key
        )
    return llm

def load_embedding_model():
    initialized_embedding_model = AzureOpenAIEmbeddings(
        openai_api_version=embedding_model_openai_api_version,
        model=embedding_model_deployment_name,
        api_key=embedding_model_azure_openai_api_key,
        azure_endpoint=embedding_model_azure_endpoint
    )
    return initialized_embedding_model