from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI


load_dotenv()

st.title("💬 Chatbot")

# session_state에 messages Key 값 지정 및 Streamlit 화면 진입 시, AI의 인사말을 기록하기
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "안녕하세요! 무엇을 도와드릴까요?"            
        }
    ]
    
# 사용자나 AI가 질문/답변 주고 받을 시, 이를 기록하는 session_state
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    
# 챗봇으로 활용할 AI 모델 선언
chat = ChatOpenAI(model="gpt-4o", temperature=0)

# chat_input()에 입력값이 있는 경우
if prompt := st.chat_input():
    # messages라는 session_state에 역할은 사용자, 콘텐츠는 프롬프트를 각각 저장
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # chat_message() 함수로 사용자 채팅 버블에 Prompt 메시지를 기록
    st.chat_message("user").write(prompt)
    
    response = chat.invoke(prompt)
    msg = response.content
    
    # messages라는 session_state에 역할은 AI, 콘텐츠는 API 답변을 각각 저장
    st.session_state.messages.append({"role": "assistant", "content": msg})
    
    # chat_message() 함수로 AI 채팅 버블에 API 답변을 기록
    st.chat_message("assistant").write(msg)