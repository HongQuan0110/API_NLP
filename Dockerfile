FROM python:3.6

RUN mkdir /code

WORKDIR /code

EXPOSE 8000

COPY . .

RUN pip3 install -r requirements.txt

# CMD [ "python3", "api/manage.py", "runserver", "0.0.0.0:8000"]
CMD python3 api/manage.py runserver 0.0.0.0:$PORT
