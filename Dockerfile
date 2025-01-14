FROM python:3.11-slim

WORKDIR /app

ADD requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

ADD ./main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]