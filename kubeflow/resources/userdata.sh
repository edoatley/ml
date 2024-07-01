#!/bin/bash
UserRunAs='azureuser'
KubernetesVersion='1.30'
KubeflowVersion='1.8'
KubeflowDashboardUsername='azureuser@example.com'
KubeflowDashboardPassword='AzureUser123!'

apt update
apt -y upgrade
for snap in juju juju-wait charmcraft; do sudo snap install $snap --classic; done
snap install microk8s --classic --channel=${KubernetesVersion}/stable
sudo snap refresh charmcraft --channel latest/candidate
usermod -a -G microk8s "${UserRunAs}"
mkdir /home/${UserRunAs}/.kube
chown -f -R "${UserRunAs}" /home/${UserRunAs}/.kube

microk8s enable dns storage metallb:"10.64.140.43-10.64.140.49,192.168.0.105-192.168.0.111"
sleep 120
microk8s.kubectl wait --for=condition=available -nkube-system deployment/coredns deployment/hostpath-provisioner
microk8s.kubectl -n kube-system rollout status ds/calico-node

su "${UserRunAs}" -c 'juju bootstrap microk8s uk8s-controller'
su "${UserRunAs}" -c 'juju add-model kubeflow'
su "${UserRunAs}" -c "juju deploy kubeflow --channel=${KubeflowVersion} --trust"

su "${UserRunAs}" -c "juju config dex-auth public-url=http://10.64.140.43.nip.io; juju config oidc-gatekeeper public-url=http://10.64.140.43.nip.io; juju config dex-auth static-username=${KubeflowDashboardUsername}; juju config dex-auth static-password=${KubeflowDashboardPassword}"
sleep 720
echo "Charmed Kubeflow deployed"

su "${UserRunAs}" -c 'juju run --unit istio-pilot/0 -- "export JUJU_DISPATCH_PATH=hooks/config-changed; ./dispatch"'
su "${UserRunAs}" -c 'juju deploy mlflow-server'
su "${UserRunAs}" -c 'juju deploy charmed-osm-mariadb-k8s mlflow-db'
su "${UserRunAs}" -c 'juju relate minio mlflow-server'
su "${UserRunAs}" -c 'juju relate istio-pilot mlflow-server'
su "${UserRunAs}" -c 'juju relate mlflow-db mlflow-server'
su "${UserRunAs}" -c 'juju relate mlflow-server admission-webhook'

echo "Charmed MlFlow deployed"