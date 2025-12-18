FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV  PYTHONUNBUFFERED=1

WORKDIR /garbage_app

RUN apt-get update && apt-get install -y --no-install-recommends \
 libsndfile1 \
 gcc \
 g++ \
 && apt=get clean && rm -rf /var/lib/apt/list/*

RUN pip install --no-cache-dir torch==2.2.0+cpu torchvision==0.17.0+cpu torchaudio==2.2.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

COPY req.txt .
RUN pip install --no-cache-dir -r req.txt

COPY main.py .
COPY model_gar.pth .

EXPOSE 8000

CMD ["uvicorn" , "main:garbage_app" , "--host" , "0.0.0.0" , "--port" , "8000" , "--workers" , "2"]

