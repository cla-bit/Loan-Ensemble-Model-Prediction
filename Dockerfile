FROM python3.11-slim

ENV PYTHONUNBUFFERED 1

# Install additional dependencies including PostgreSQL development libraries
RUN apt-get update
RUN python3 -m pip install pip --upgrade

WORKDIR /usr/src/app

COPY ./requirements.txt ./

#Added these two lines of COPY below
COPY ./static /app/staticfiles
#COPY ./media /app/media

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "python manage.py collectstatic --noinput && python manage.py runserver 0.0.0.0:8000"]
