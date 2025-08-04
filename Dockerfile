FROM python:3.10

RUN apt-get -y update


# set display port to avoid crash
ENV DISPLAY=:99

# upgrade pip
RUN pip install --upgrade pip


RUN python -m pip install --upgrade pip

COPY requirements.txt requirements.txt
RUN python -m pip install -r requirements.txt

COPY . .

CMD ["/bin/bash", "+x", "/entrypoint.sh"]