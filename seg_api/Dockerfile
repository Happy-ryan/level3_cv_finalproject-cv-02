FROM python:3.9.18
WORKDIR /project
COPY . /project
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD python app/main.py