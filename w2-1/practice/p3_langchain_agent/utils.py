import os
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from datetime import datetime
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun, YouTubeSearchTool
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
import tempfile


def load_model(model_name):
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model_name=model_name)
    return llm 


def load_prompt(case):
    with open(f"prompts/{case}.prompt", "r", encoding="utf-8") as file:
        prompt = file.read().strip()
    return prompt


def set_memory():
    return ConversationBufferMemory(memory_key="chat_history", return_messages=True)


def create_vectordb(uploaded_files):
    doc_paths = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            doc_paths.append(temp_file.name)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
    chunks = []

    for path in doc_paths:
        loader = PyPDFLoader(path)
        chunks.extend(loader.load_and_split(splitter))

    embeddings = OpenAIEmbeddings() 
    vectordb = FAISS.from_documents(chunks, embeddings)
    
    return vectordb


def create_agent(llm, vectordb, memory):
    ddg_search = DuckDuckGoSearchRun()
    youtube_search = YouTubeSearchTool()

    if vectordb:
        retriever = vectordb.as_retriever()
        file_search = create_retriever_tool(retriever, name="file_search", description="Searches content within the uploaded files.")
        tools = [ddg_search, youtube_search, file_search]
        system_prompt = load_prompt('system2')
    else:
        tools = [ddg_search, youtube_search]
        system_prompt = load_prompt('system1')

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template=system_prompt)),
            MessagesPlaceholder(variable_name='chat_history', optional=True),
            HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template="{input}")),
            MessagesPlaceholder(variable_name='agent_scratchpad')
        ]
    )

    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory)
    
    return agent_executor


def invoke_agent(agent_executor, user_input):
    today = datetime.today().strftime('%Y-%m-%d')
    full_input = f"Today's date is {today}. {user_input}"
    response = agent_executor.invoke({'input': full_input})
    return response["output"]
