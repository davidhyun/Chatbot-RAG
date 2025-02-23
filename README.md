# 챗봇 만들기. 그런데 RAG를 곁들인

## 1. RAG (Retrieval-Augmented Generation)
'검색 증강 생성'은 사실이 아닌데도 사실처럼 답변하는 LLM의 환각 문제를 효과적으로 해결할 수 있는 기법이다.
LLM이 사실에 근거한 답변을 하도록 외부지식을 참고한다. 이 외부지식은 문서를 수치화(임베딩)하여 벡터 DB에 저장하여 답변에 활용한다.

## 2. Closed LLM VS OpenSource LLM
|구분|Closed LLM|OpenSource LLM|
|비용|사용량 기반 종량제|LLM 구축 및 추론 비용|
|성능|높음|중간~낮음|
|보안|중간|높음|
|맞춤화|일부 파인튜닝만 지원|모델 구조 변경 통한 성능 향상|
|예시|ChatGPT(OpenAI)|Llama(Meta)|

## 3. 개발환경
- Python, Streamlit, LangChain, OpenAI
```bash
$ streamlit run <app.py>
```