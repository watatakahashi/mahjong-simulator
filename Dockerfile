FROM jupyter/datascience-notebook

WORKDIR /home/jovyan/

ADD . /home/jovyan/

RUN pip install -r requirements.txt

WORKDIR /home/jovyan/work