import streamlit as st
import time
from langchain_core.messages import HumanMessage, AIMessage
from utils import load_model, set_memory, initialize_chain, generate_message


st.title("페르소나 챗봇")
st.markdown("<br>", unsafe_allow_html=True)

character_name = st.selectbox("**캐릭터를 골라줘!**", 
                              ("buddha", "lucky_vicky"), 
                              index=0, 
                              key="character_name_select")

st.session_state.character_name = character_name

model_name = st.selectbox("**모델을 골라줘!**", 
                          ("gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"), 
                          index=0, 
                          key="model_name_select")

st.session_state.model_name = model_name

if "chat_started" not in st.session_state:
    st.session_state.chat_started = False
    st.session_state.memory = None
    st.session_state.chain = None

def start_chat():
    llm = load_model(st.session_state.model_name)
    st.session_state.chat_started = True
    st.session_state.memory = set_memory()
    st.session_state.chain = initialize_chain(llm, st.session_state.character_name, st.session_state.memory)

if st.button("Start Chat"):
    start_chat()

st.markdown("<br>", unsafe_allow_html=True)

if st.session_state.chat_started:
    if st.session_state.memory is None or st.session_state.chain is None:
        start_chat()

    for message in st.session_state.memory.chat_memory.messages:
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            continue
        with st.chat_message(role):
            st.markdown(message.content)

    if prompt := st.chat_input():
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            response_content = generate_message(st.session_state.chain, prompt)

            for chunk in response_content.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response.strip())