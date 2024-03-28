# CLIP: From Local to Nautilus Tutorial

## Setup:
Docker:
1. Install Docker (https://docs.docker.com/engine/install/)
2. Create DockerHub account (https://hub.docker.com)

Kubernetes:
1. Install kubectl (https://docs.nationalresearchplatform.org/userdocs/start/quickstart/)
2. Request/use namespace in Nautilus for your project (https://docs.nationalresearchplatform.org/userdocs/start/get-access/)

## Start building docker image locally:

### 1. Build the image using Dockerfile
- Empty image (only Jupyter is installed):
`docker build . -f Dockerfile -t segurvich/ecoviz_abc:latest --platform=linux/amd64`
### 2. Test the image locally:
`docker run -p 8888:8888 segurvich/ecoviz_abc`

### 3. Push the image to DockerHub (need to have account)
`docker push segurvich/ecoviz_abc`

## Now lets bring everything to Nautilus:
_Documentation: https://docs.nationalresearchplatform.org/userdocs/jupyter/jupyter-pod/_

### 1. Create pod in Nautilus using EcoViz_CLIP.yaml:
`kubectl create -f  EcoViz_AB_classifier.yaml`

### 2. List current pods and their status:
`kubectl get pods`

### 3. Describe our pod (get info):
`kubectl describe pod sergey-abc`

### 4. Once created, we can get logs:
`kubectl logs sergey-abc`

### 5. Once created, we can go inside the pod:
`kubectl exec -it sergey-abc bash`

### 6. Setup port forwarding to access Jupyter Lab from local browser:
`kubectl port-forward sergey-abc 8888:8888`

### 7. Access the Jupyter Hub link from the logs using your local browser
http://127.0.0.1:8888/lab/tree/EcoViz_CLIP/EcoViz_CLIP_model.ipynb

### 8. Clone repo:
- `git clone https://github.com/sergeygurvich/AB_classifier`
- 
### 9. Add data, modify paths in config file, if needed

### 10. Tear down:
`kubectl delete -f  EcoViz_AB_classifier.yaml`
