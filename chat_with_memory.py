import os
from dotenv import load_dotenv
import uuid
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory


load_dotenv()

# cache_resource로 한번 실행한 결과 캐싱해두기
@st.cache_resource
def load_and_split_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()

# 텍스트 청크들을 Chroma 안에 임베딩 벡터로 저장
@st.cache_resource
def create_vector_store(_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(_docs)
    vectorstore = Chroma.from_documents(split_docs, OpenAIEmbeddings(model='text-embedding-3-small'))
    return vectorstore

# 만약 기존에 저장해둔 ChromaDB가 있는 경우, 이를 로드
@st.cache_resource
def get_vector_store(_docs):
    persist_directory = "./chroma_db"
    if os.path.exists(persist_directory):
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=OpenAIEmbeddings(mode='text-embedding-3-small')
        )
    else:
        return create_vector_store(_docs)
    
    
# PDF 문서 로드-벡터 DB 저장-검색기-히스토리 모두 합친 Chain 구축
@st.cache_resource
def initialize_components(selected_model):
    file_path = r"data/대한민국헌법(헌법)(제00010호)(19880225).pdf"
    pages = load_and_split_pdf(file_path)
    
    vectorstore = get_vector_store(pages)
    retriever = vectorstore.as_retriever()
    
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
st.markdown(
    """
    <style>
        body {
            background-color: #f0f2f6; /* Change this to any color */
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("헌법 Q&A 챗봇 💬")
selected_model = st.selectbox("Select GPT Model", ("gpt-4o", "gpt-3.5-turbo-0125"))
rag_chain = initialize_components(selected_model)
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
            "content": "헌법에 대해 무엇이든 물어보세요!"
        }
    ]
    
for msg in chat_history.messages:
    st.chat_message(msg.type).write(msg.content)
    
if prompt_message := st.chat_input("Yourt question"):
    st.chat_message("human").write(prompt_message)
    with st.chat_message("ai"):
        with st.spinner("두뇌 풀가동 중..."):
            config = {"configurable": {"session_id": "any"}}
            response = conversational_rag_chain.invoke(
                {"input": prompt_message},
                config
            )
            
            answer = response["answer"]
            st.write(answer)
            with st.expander("참고 문서 확인"):
                for doc in response["context"]:
                    st.markdown(doc.metadata["source"], help=doc.page_content)