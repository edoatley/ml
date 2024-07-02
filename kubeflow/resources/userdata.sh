#!/bin/bash
# This is a basic installation script that will install minikube and kubeflow on a VM

# Install docker
apt update -y && apt upgrade -y
sudo apt update -y
sudo apt install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker
sudo apt install -y apt-transport-https ca-certificates curl
curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://apt.kubernetes.io/ kubernetes-xenial main" | sudo tee /etc/apt/sources.list.d/kubernetes.list

# Install CLI tools kubectl and kustomize
sudo snap install kubectl --classic
sudo snap install kustomize

# Install minikube
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
minikube start --driver=docker
minikube status

git clone https://github.com/kubeflow/manifests.git ~/tmp/kubeflow
cd ~/tmp/kubeflow
while ! kustomize build example | awk '!/well-defined/' | kubectl apply -f -; do echo "Retrying to apply resources"; sleep 10; done
# rm -rf ~/tmp/kubeflow # could clean up with this
