FROM python:3.10.12-slim

WORKDIR /app

COPY Pipfile ./
COPY Pipfile.lock ./

RUN python -m pip install --upgrade pip
RUN pip install pipenv && pipenv install --dev --system --deploy

CMD ["streamlit", "run", "chatapp.py"]
