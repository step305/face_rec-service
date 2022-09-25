Download from Google Disk face_rec-image.zip
Download from Google Disk users_db-image.zip

docker load -i users_db-image.zip
docker load -i face_rec-image.zip

before start - download user_base.db
in start.ba change path to folder where user_base.db is stored

docker run -d -p 8000:5000 -v CHANGE_HERE:/backend/db_media --name users_db-service users_db-service:v0.1.0

start.bat

train.py - example of how to add users
test_rec.py - example of how t recognize photos
