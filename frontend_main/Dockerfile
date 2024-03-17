FROM python:3.9.18
WORKDIR /project
COPY . /project

RUN pip install --upgrade pip
RUN pip install -r frontend_requirements.txt

CMD streamlit run Home.py --server.enableXsrfProtection=false
