#!/bin/bash
UserRunAs='azureuser'
KubernetesVersion='1.30'
KubeflowVersion='1.8'
KubeflowDashboardUsername='azureuser@example.com'
KubeflowDashboardPassword='AzureUser123!'

# Package management
apt update
apt -y upgrade

# Install Minikube
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
alias kubectl="minikube kubectl --"

curl -s https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh  | bash
sudo install kustomize /usr/local/bin/kustomize


git clone https://github.com/kubeflow/manifests.git
cd manifests
while ! kustomize build example | awk '!/well-defined/' | kubectl apply -f -; do echo "Retrying to apply resources"; sleep 10; done

