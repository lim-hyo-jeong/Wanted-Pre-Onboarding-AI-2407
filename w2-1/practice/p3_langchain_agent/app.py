import streamlit as st
import time
from langchain_core.messages import HumanMessage, AIMessage
from utils import load_model, set_memory, create_vectordb, create_agent, invoke_agent

st.title("📊금융 상담 에이전트")
st.markdown("<br>", unsafe_allow_html=True)

model_name = st.selectbox("**모델을 골라주세요.**", 
                          ("gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"), 
                          index=0, 
                          key="model_name_select")

st.session_state.model_name = model_name

if "chat_started" not in st.session_state:
    st.session_state.chat_started = False
    st.session_state.memory = None
    st.session_state.vectordb = None 
    st.session_state.agent = None
    st.session_state.uploaded_files = None

def start_chat(uploaded_files):
    llm = load_model(st.session_state.model_name)
    st.session_state.chat_started = True
    st.session_state.memory = set_memory()
    if uploaded_files:
        st.session_state.vectordb = create_vectordb(uploaded_files)
    else:
        st.session_state.vectordb = None 
    st.session_state.agent = create_agent(llm, st.session_state.vectordb, st.session_state.memory)

def config_change(uploaded_files):
    llm = load_model(st.session_state.model_name)
    st.session_state.vectordb = create_vectordb(uploaded_files)
    st.session_state.agent = create_agent(llm, st.session_state.vectordb, st.session_state.memory)

with st.sidebar:
    st.header("분석할 기업의 연간 보고서/분기 보고서를 업로드해주세요.")
    uploaded_files = st.file_uploader("Upload your documents", accept_multiple_files=True, type=["pdf"])

    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        with st.spinner('잠시 기다려주세요 ...🧚‍♀️'):
            config_change(uploaded_files)

if st.button("Start Chat"):
    with st.spinner('잠시 기다려주세요 ...🧚‍♀️'):
        start_chat(st.session_state.uploaded_files)

st.markdown("<br>", unsafe_allow_html=True)

if st.session_state.chat_started:
    if st.session_state.memory is None or st.session_state.agent is None:
        start_chat(st.session_state.uploaded_files)

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
            response_content = invoke_agent(st.session_state.agent, prompt)
            response_content = response_content.replace('$', '\\$')

            for line in response_content.split('\n'):
                for chunk in line.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")
                full_response += '\n'

            message_placeholder.markdown(full_response.strip())

