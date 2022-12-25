#test deploy_03.04
FROM python:3.8
RUN rm -rf /usr/share/tesseract-ocr
RUN echo $(ls -1 /usr/share)
RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev
RUN export PATH=/user/local/bin:$PATH
RUN apt-get update && apt-get install libleptonica-dev libtesseract-dev libpango1.0-dev autoconf automake ca-certificates g++ git libtool libleptonica-dev make pkg-config libpng-dev libjpeg62-turbo-dev libtiff5-dev zlib1g-dev libsdl-pango-dev libicu-dev libcairo2-dev bc ffmpeg libsm6 libxext6 -y
RUN wget https://github.com/tesseract-ocr/tesseract/archive/4.1.3.zip -P /usr/share/
RUN cd /usr/share/ && unzip 4.1.3.zip
RUN mkdir -p /usr/share/tesseract-ocr
RUN mv /usr/share/tesseract-4.1.3/* /usr/share/tesseract-ocr
RUN rm -rf /usr/share/tesseract-4.1.3

#RUN echo $(ls -1 /usr/share/tesseract-ocr)
RUN cd /usr/share/tesseract-ocr && ./autogen.sh && ./configure && make && make install && ldconfig && make training && make training-install && tesseract --version

# Installing the custom repo files
RUN rm -rf /usr/share/tesseract-ocr/tessdata/*
RUN git clone --depth 1 https://github.com/tesseract-ocr/tessdata_fast /usr/share/tesseract-ocr/tessdata

### Installing python libraries
RUN pip install --upgrade pip
RUN pip install python-json-logger==2.0.2
RUN pip install psutil==5.8.0
RUN pip install Pillow==8.4.0
RUN pip install opencv-contrib-python==4.5.4.60
RUN pip install pytesseract==0.3.8
RUN pip install scikit-image==0.18.3
RUN pip install numpy==1.21.4
RUN pip install deskew==0.10.39
RUN pip install boto3==1.20.13
RUN pip install PyYAML==6.0
RUN pip install ray==1.9.0
RUN pip install arrow==1.2.2
#RUN pip install -r requirements.txt

### Copy the source main.py file
COPY ./main.py ./crop.py ./Server.py ./tools.py /opt/

### Expose the port and run the application
EXPOSE 8080
CMD [ "python", "/opt/main.py" ]