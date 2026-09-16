FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements-app.txt ./
RUN pip install --no-cache-dir -r requirements-app.txt
COPY code/deployment/app ./code/deployment/app
EXPOSE 8501
CMD ["streamlit", "run", "code/deployment/app/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
