docker run -d -p 8000:5000 -v e:\:/backend/db_media --name users_db-service users_db-service:v0.1.0
timeout 5

docker run -d -p 5000:5000 --name face_rec-service face_rec-service:v0.1.0

docker network create face_rec_network

docker network connect face_rec_network users_db-service

docker network connect face_rec_network face_rec-service

docker stop face_rec-service
docker start face_rec-service
