from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS # Vectorstore 라이브러리
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent

loader = PyPDFLoader(r"data/★육아휴직제도 사용안내서_배포.pdf")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
    ).split_documents(docs)

vector = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vector.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "parental_leave",
    "육아 휴직과 관련한 정보를 검색합니다. 육아 휴직관련한 질문이 입력되면 이 도구를 사용합니다."
)

search_tool = TavilySearchResults() # Tavily: 인터넷 검색 엔진

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [search_tool, retriever_tool]
prompt = hub.pull("hwchase17/openai-functions-agent")
agent = create_openai_tools_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

queries = [
    "육아 휴직은 어떻게 사용할 수 있어?",
    "나스닥에 상장된 애플의 오늘 주가를 알려줘"
]
for q in queries:
    agent_executor.invoke({"input": q})