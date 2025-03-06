import os
import tempfile
import shutil
import streamlit as st
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS # Vectorstore 라이브러리
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent

load_dotenv()

# Initialization
# if '' not in st.session_state:

########## [요구사항] ##########
# 1. 채팅방 초기화 기능 (Required)
#   - 단일 사용자 사용
#   - 사용자 인증 및 로그인 기능 없음
# 2. 문서 기반 답변 (Required)
#   - PDF 업로드 기능
# 3. 채팅 이력을 고려하여 답변 (Required)
# 4. 필요시 인터넷 검색 (Optional)
#   - 에이전트 구현
#   - 인터넷 검색 Tool 사용

########## [Process Sequence] ##########


# cache_resource로 한번 실행한 결과 캐싱해두기
# @st.cache_resource
def load_and_split_pdf(_file):
    # 임시 파일을 생성하여 업로드된 PDF 파일의 데이터를 저장
    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmp_file:
        # 업로드된 파일의 내용을 임시 파일에 기록
        tmp_file.write(_file.getvalue())
        
        # 임시 파일의 경로를 변수에 저장
        tmp_file_path = tmp_file.name
        
        # 임시 파일의 데이터를 로드하고 페이지를 분할
        loader = PyPDFLoader(file_path=tmp_file_path)
        pages = loader.load_and_split()
        
        return pages # 분할된 페이지들을 반환
    
# 텍스트 청크들을 Chroma 안에 임베딩 벡터로 저장
# @st.cache_resource
def create_vectorstore(documents, persist_directory):
    vectorstore = Chroma.from_documents(
        documents,
        OpenAIEmbeddings(model='text-embedding-3-small'),
        persist_directory=persist_directory,
        client_settings=Settings(allow_reset=True)
    )
    return vectorstore

# 만약 기존에 저장해둔 ChromaDB가 있는 경우, 이를 로드
# @st.cache_resource
def get_vectorstore(persist_directory):
    if os.path.exists(persist_directory):
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=OpenAIEmbeddings(model='text-embedding-3-small'),
        )
        return vectorstore
    else:
        return None

def init_vectorstore(persist_directory):
    # 벡터스토어 초기화
    client = chromadb.PersistentClient(
        path=persist_directory,
        settings=Settings(allow_reset=True)
        )
    client.reset()
    client.clear_system_cache()

# 프롬프트 | LLM 모델 | 검색기 RAG 체인 구축
# @st.cache_resource
def chaining(retriever, selected_model):
    # 채팅 히스토리 요약 시스템 프롬프트 (주어진 채팅 이력과 사용자의 마지막 질문을 바탕으로 하나의 독립된 질문을 구성)
    contextualize_q_system_prompt = """
        Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is.
    """
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    # 질문-답변 시스템 프롬프트
    qa_system_prompt = """
        You are an assistant for question-answering tasks. \
        Use the following pieces of retrieved context to answer the question. \
        If you don't know the answer, just say that you don't know. \
        Keep the answer perfect. please use imogi with the answer. \
        대답은 한국어로 하고, 존댓말을 써줘. \

        {context}
    """
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    llm = ChatOpenAI(model=selected_model)
    
    # 채팅 이력을 고려하여 질문과 유사한 문서를 검색하는 체인
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain


# Streamlit UI
st.title("PDF Q&A 챗봇 💬")

# Initialize vector store
persist_directory = "./chroma_db"
init_vectorstore(persist_directory)

if st.button("채팅방 초기화"):
    # 벡터DB 초기화
    init_vectorstore(persist_directory)

    # 채팅 이력 세션 초기화   
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "안녕하세요. PDF 내용에 대해 무엇이든 물어보세요!"
        }
    ]
    st.session_state["chat_messages"] = []
        
    # 캐시 초기화
    st.cache_data.clear()
    st.rerun()

selected_model = st.selectbox("Select GPT Model", ("gpt-4o", "gpt-3.5-turbo-0125"))
uploaded_file = st.file_uploader("PDF 파일을 업로드해주세요.", type=["pdf"])

if uploaded_file is not None:
    # PDF 파일을 로드하고 페이지로 분할
    pages = load_and_split_pdf(uploaded_file)
    
    # 페이지 문서를 텍스트 청크로 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", "."])
    splitted_docs = text_splitter.split_documents(pages)
    
    # 벡터스토어 구성
    vectorstore = create_vectorstore(splitted_docs, persist_directory)
    retriever = vectorstore.as_retriever()
    
    # RAG 체인 구축
    rag_chain = chaining(retriever, selected_model)

    chat_history = StreamlitChatMessageHistory(key="chat_messages")

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id : chat_history,
        input_messages_key = "input",
        history_messages_key = "chat_history",
        output_messages_key = "answer",
    )

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": "안녕하세요. PDF 내용에 대해 무엇이든 물어보세요!"
            }
        ]
        
    for msg in chat_history.messages:
        st.chat_message(msg.type).write(msg.content)
        
    if prompt_message := st.chat_input("Your question"):
        st.chat_message("human").write(prompt_message)
        with st.chat_message("ai"):
            config = {"configurable": {"session_id": "any"}}
            with st.spinner("두뇌 풀가동 중..."):
                response = conversational_rag_chain.invoke(
                    input={"input": prompt_message},
                    config={"configurable": {"session_id": "any"}}
                )
                
                # 답변 및 참고문서 출력
                st.write(response["answer"])
                with st.expander("참고 문서 확인"):
                    for doc in response["context"]:
                        st.markdown(f"### {uploaded_file.name}")
                        st.markdown(f"#### {doc.metadata['page_label']} Page")
                        st.write(doc.page_content)