FROM jupyter/pyspark-notebook
COPY . .
RUN python3 -m pip install -r requirements.txt
USER root

CMD ["python3", "consume_store.py"]