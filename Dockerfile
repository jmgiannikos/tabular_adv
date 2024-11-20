# use tabularbench as root
FROM python:3.8-buster

# install TALENT dependencies
# SKIPPED

RUN pip install tabularbench
RUN pip install notebook
RUN pip install ipywidgets

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
