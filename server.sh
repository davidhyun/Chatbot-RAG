#!/bin/bash

STATUS=$1

IMAGE_NAME="llm-chatbot:v0.1"
CONTAINER_NAME="llm-chatbot"

if [ "$STATUS" = "run" ]; then
    echo "Executing: docker run"
    docker run -dit \
        --name $CONTAINER_NAME \
        -p 18502:8501 \
        -v $PWD:/app \
        --env-file .env \
        $IMAGE_NAME \
        streamlit run chat_with_memory_deploy.py

elif [ "$STATUS" = "start" ]; then
    echo "Executing: docker start"
    docker start $CONTAINER_NAME

elif [ "$STATUS" = "stop" ]; then
    echo "Executing: docker stop"
    docker stop $CONTAINER_NAME

elif [ "$STATUS" = "restart" ]; then
    echo "Removing VectorDB and Restarting Container"
    sudo rm -rf "$PWD/chroma_db"
    docker restart $CONTAINER_NAME

else
    echo "Usage: ./$0 {run|start|stop|restart}"
    exit 1
fi
