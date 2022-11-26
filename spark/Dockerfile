FROM jupyter/pyspark-notebook
COPY . .
RUN python3 -m pip install -r requirements.txt
USER root

#RUN ["python3", "consume_store.py"]


#RUN echo '[...] Preprocessing data'
#RUN python3 preprocess.py
