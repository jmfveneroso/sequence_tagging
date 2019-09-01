FROM tensorflow/tensorflow:latest-gpu-py3
# FROM lspvic/tensorboard-notebook

RUN apt-get update     && \
    apt-get upgrade -y && \
    apt-get install -y git

ENV PYTHONUNBUFFERED 1
RUN mkdir /code
WORKDIR /code

# Add Jovyan to sudoers.
# RUN chown jovyan:users /code
# RUN echo "jovyan ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

RUN pip install --upgrade pip

# USER jovyan

ADD requirements.txt /code/
RUN pip install -r requirements.txt

# Set password to jupyter notebook. Default is "latin".
RUN mkdir /root/.jupyter
RUN echo "c.NotebookApp.password='sha1:b4b547d15cb6:5bc10ecee9305d8120678c593e5b219614363650'">>/root/.jupyter/jupyter_notebook_config.py

CMD jupyter notebook --no-browser --ip 0.0.0.0 --port 8888 /code --allow-root


# RUN echo "c.NotebookApp.password='sha1:b4b547d15cb6:5bc10ecee9305d8120678c593e5b219614363650'">>/home/jovyan/.jupyter/jupyter_notebook_config.py

# ENTRYPOINT ["/code/entrypoint.sh"]
