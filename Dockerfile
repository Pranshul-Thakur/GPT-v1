FROM python:3.9

#creation of a virtual directory to run the docker image
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
#copies everything in the docker image w

CMD ["python", "BPE.py"]

#works on the basis of layer caching