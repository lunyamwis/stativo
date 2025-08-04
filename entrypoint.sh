#!/bin/bash

# source /root/.local/share/virtualenvs/brooks-insurance-*/bin/activate

echo "<<<<<<<< Collect Staticfiles>>>>>>>>>"
python3 manage.py collectstatic --noinput


sleep 5
echo "<<<<<<<< Database Setup and Migrations Starts >>>>>>>>>"
# Run database migrations
# python3 manage.py makemigrations
python3 manage.py migrate &

sleep 5
echo "<<<<<<< Initializing the Database >>>>>>>>>>"
echo " "
python manage.py loaddata initialization.yaml
echo " "
echo "<<<<<<<<<<<<<<<<<<<< START Celery >>>>>>>>>>>>>>>>>>>>>>>>"

# # start Celery worker
celery -A eLMS worker --loglevel=info &

# # start celery beat
celery -A eLMS beat --loglevel=info &

sleep 5
echo "<<<<<<<<<<<<<<<<<<<< START API >>>>>>>>>>>>>>>>>>>>>>>>"
# python manage.py runserver 0.0.0.0:8000
# python manage.py runserver 

# Start the API with gunicorn
gunicorn --bind 0.0.0.0:8000 eLMS.wsgi --reload --access-logfile '-' --workers=2