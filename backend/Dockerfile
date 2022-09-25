FROM python:3.9-buster
WORKDIR /backend
COPY requirements.txt /backend
RUN pip3 install cmake
RUN pip3 install buildtools
RUN apt-get update -y
RUN apt install libgl1-mesa-glx -y
RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y
RUN pip3 install --upgrade pip -r requirements.txt
RUN pip3 install --upgrade pip -r requirements.txt
COPY . /backend
EXPOSE 5000
ENTRYPOINT ["python3"]
CMD ["app.py", "-p", "5000"]
