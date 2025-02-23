import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

#Chroma tenant 오류 방지 위한 코드
import chromadb
chromadb.api.client.SharedSystemClient.clear_system_cache()

# Streamlit은 사용자와 상호작용할 때마다 전체 파이썬 스크립트를 재실행하여 속도가 느려질 수 있음
# 이를 방지하기 위해 데코레이터 '@st.cache_resource'를 사용하여 웹앱이 구동될 때 생성된 데이터를 캐싱하는 방법을 사용

# PDF 파일 로드 및 분할
@st.cache_resource
def load_and_split_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()

# 텍스트 청크들을 Chroma(VectorDB) 안에 임베딩 벡터로 저장
@st.cache_resource
def create_vector_store(_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_docs = text_splitter.split_documents(_docs)
    persist_directory = "./chroma_db"
    vectorstore = Chroma.from_documents(
        split_docs,
        OpenAIEmbeddings(model='text-embedding-3-small'),
        persist_directory=persist_directory
    )
    return vectorstore

# 기존에 저장해둔 ChromaDB가 있는 경우, 이것을 로드
@st.cache_resource
def get_vector_store(_docs):
    persist_directory = "./chroma_db"
    if os.path.exists(persist_directory):
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=OpenAIEmbeddings(model='text-embedding-3-small')
        )
    else:
        return create_vector_store(_docs)
    
# 검색된 문서를 하나의 텍스트로 합치는 헬퍼 함수
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@st.cache_resource
def chaining():
    # 기반 지식(문서) 주입
    #file_path = r"data/대한민국헌법(헌법)(제00010호)(19880225).pdf"
    file_path = "data/(제2023-269호)+산정특례제도+질의응답.pdf"
    
    # 문서 분할    
    pages = load_and_split_pdf(file_path)
    
    # 문서를 수치화(임베딩)하여 벡터 DB에 저장
    vectorstore = get_vector_store(pages)

    # 벡터 DB를 검색기로 선언
    retriever = vectorstore.as_retriever()
    
    qa_system_prompt = """
        You are an assistant for question-answering tasks. \
        Use the following pieces of retrieved context to answer the question. \
        If you don't know the answer, just say that you don't know. \
        Keep the answer perfect. please use imogi with the answer.
        Please answer in Korean and use respectful language.\
        {context}
    """
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("human", "{input}"),
        ]
    )
    
    # 언어 모델 선언
    llm = ChatOpenAI(model='gpt-4o')
    
    # RAG 체인
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain


# Streamlit UI
st.title("산정특례 Q&A 챗봇 💬")

rag_chain = chaining()

# session_state에 messages Key 값 지정 및 Streamlit 화면 진입 시, AI의 인사말을 기록하기
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "안녕하세요. 산정특례에 대해 무엇이든 물어보세요!"            
        }
    ]
    
# 사용자나 AI가 질문/답변 주고 받을 시, 이를 기록하는 session_state
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt_message := st.chat_input("질문을 입력해주세요 :"):
    st.chat_message("human").write(prompt_message)
    st.session_state.messages.append({"role": "user", "content": prompt_message})
    with st.chat_message("ai"):
        with st.spinner("두뇌 풀가동 중..."):
            response = rag_chain.invoke(prompt_message)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
    
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