Download from Google Disk face_rec-image.zip

!!!! if not working as is - install Docker Desktop !!!!
install Docker Desktop on Windows (before enable Hyper V in add/remove Windows components,
 Virtual Machine Platform and Windows Subsystem for Linux -
 https://stackoverflow.com/questions/66267529/docker-desktop-3-1-0-installation-issue-access-is-denied
 )
run docker desktop from start menu
!!!!!


!!!!!!!!!!!!! Use container !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
On server load and run:
docker load -i face_rec-image.zip
docker run -d -p 5000:5000 --name face_rec-service face_rec-service:v0.1.0
docker network create face_rec_network
!!! users_db-service should be ready and running
docker network connect face_rec_network users_db-service
docker network connect face_rec_network face_rec-service


Stop service:
docker stop face_rec-service

Remove service:
docker rm -f face_rec-service
docker rmi --force face_rec-service:v0.1.0


!!!!!!!!!!!!! For dev - build and deploy !!!!!!!!!!!!!!!!
Build container:
docker build -t face_rec-service:v0.1.0 backend/

run container:
docker run -d -p 5000:5000 --name face_rec-service face_rec-service:v0.1.0

stop container:
docker stop face_rec-service
docker rm -f face_rec-service

remove container:
docker rmi --force face_rec-service:v0.1.0

Save to file
docker save -o face_rec-image.zip face_rec-service
