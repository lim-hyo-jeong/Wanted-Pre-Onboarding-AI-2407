import os
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory


def load_model(model_name):
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model_name=model_name)
    return llm 


def load_prompt(character_name):
    with open(f"prompts/{character_name}.prompt", "r", encoding="utf-8") as file:
        prompt = file.read().strip()
    return prompt


def set_memory():
    return ConversationBufferMemory(memory_key="chat_history", return_messages=True)


def initialize_chain(llm, character_name, memory):
    system_prompt = load_prompt(character_name)
    custom_prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )
    chain = LLMChain(llm=llm, prompt=custom_prompt, verbose=True, memory=memory)
    return chain


def generate_message(chain, user_input):
    result = chain({"input": user_input})
    response_content = result["text"]
    return response_content
