#!/bin/bash

docker run -dit \
    --name llm-chatbot \
    -p 18502:8501 \
    -v $PWD:/app \
    --env-file .env \
    llm-chatbot:v0.1
