# Analysis of OCT Images to detect retinopathy ENESJ - IMO

Description: This repository contains the source code, libraries and documentation for setting up the web server using Keras TF and Sklearn


## Docker configruration

0. Install libraries 
+ `pip install notebook`
+ `pip install random2 opencv-python numpy pandas matplotlib seaborn sklearn scipy sklearn keras scikit-image tqdm tensorflow imblearn`
+ ```bash pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html ```

### Increase size of docker containers 

1. Go to `/etc/containers` and edit `storage.conf` and add:   
  + `basesize = "200G"`

2. Restart docker service 
  ´systemctl restart podman´


#### Run docker container using a volume


1. Create a volume

+ `docker volume create jupyterhub-vol`
+ `docker volume ls`

2. Run docker container using a volume

+ `docker pull jupyterhub/jupyterhub`

+ `docker run -d -p 8000:8000 --name jupyterhub --mount type=volume,source=jupyterhub-vol,target=/volume jupyterhub:latest`
+ `docker exec -it jupyterhub bash`

#### Remove docker image and volume

Remove image and container

+ `docker rm <image_id>`

Remove volume

+ `docker volume rm <volume_id>`

#### Stop docker container

`docker stop <container_id>`

## See docker logs
`docker logs jupyterhub`
