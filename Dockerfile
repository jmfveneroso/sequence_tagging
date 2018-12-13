FROM tensorflow/tensorflow:latest-py3

RUN apt-get update     && \
    apt-get upgrade -y && \
    apt-get install -y git

ENV PYTHONUNBUFFERED 1
RUN mkdir /code
WORKDIR /code
ADD requirements.txt /code/
RUN pip install -r requirements.txt
# ADD . /code/

# Set password to jupyter notebook. Default is "latin".
RUN echo "c.NotebookApp.password='sha1:b4b547d15cb6:5bc10ecee9305d8120678c593e5b219614363650'">>/root/.jupyter/jupyter_notebook_config.py
