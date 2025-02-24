from dotenv import load_dotenv
import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
    
# Streamlit에서 업로드하면 파일 자체를 저장하기 때문에 임시 파일에 내용을 저장하고 파일의 경로를 반환하는 우회 방법을 사용
@st.cache_resource
def load_pdf(_file):
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
    
# 텍스트 청크들을 Chroma(VectorDB) 안에 임베딩 벡터로 저장
@st.cache_resource
def create_vector_store(_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", " ", ""])
    split_docs = text_splitter.split_documents(_docs)
    persist_directory = "./chroma_db"
    vectorstore = Chroma.from_documents(
        split_docs,
        OpenAIEmbeddings(model='text-embedding-3-small'),
        persist_directory=persist_directory
    )
    return vectorstore

# 검색된 문서를 하나의 텍스트로 합치는 헬퍼 함수
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
    

# PDF 문서기반 RAG 체인 구축
@st.cache_resource
def chaining(_pages):
    vectorstore = create_vector_store(_pages)
    retriever = vectorstore.as_retriever()
    
    qa_system_prompt = """
        질문-답변 업무를 돕는 보조원입니다. 
        질문에 답하기 위해 문서를 기반으로 검색하여 답변하세요.
        답을 모르면 모른다고 말하세요.
        한국어로 정중하게 답변하세요.
        간략하게 답변하세요.

        ## 답변 예시
        **[답변]**
        답변 내용 서술
        \n
        **[문서 내 출처 위치]**
        문서 내 출처 페이지와 정보 위치 서술

        {context}
    """

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("human", "{input}"),
        ]
    )
    
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain


# Streamlit UI
st.title("PDF 챗봇 💬")

uploaded_file = st.file_uploader("PDF 파일을 업로드해주세요.", type=["pdf"])
if uploaded_file is not None:
    pages = load_pdf(uploaded_file)

    rag_chain = chaining(pages)

    # session_state에 messages Key 값 지정 및 Streamlit 화면 진입 시, AI의 인사말을 기록하기
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": "안녕하세요. 무엇이든 물어보세요!"            
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