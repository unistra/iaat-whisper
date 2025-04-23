#!/bin/bash

docker build -f Dockerfile -t whisper-app .
docker save -o dist/whisper-app.tar whisper-app:latest