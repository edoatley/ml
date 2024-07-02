#!/bin/bash

# VM details
RESOURCE_GROUP=rg-edo-sandbox
VNET_NAME=vnet-edo-sandbox
SANDBOX_SUBNET=snet-sandbox
VM_NAME=vm-edo-sandbox
VM_SIZE=Standard_D8s_v3
VM_IMAGE=Ubuntu2204
VM_SSH_KEY=~/.ssh/id_rsa
VM_ADMIN_USER="azureuser"
VM_USER_DATA="cloud-init.yaml"

# Delete the VM and wait for it to be gone
az vm delete -g $RESOURCE_GROUP -n $VM_NAME --yes
az vm wait --deleted -g $RESOURCE_GROUP -n $VM_NAME

sleep 120

# Recreate the VM and wait for it to be running
az vm create --resource-group $RESOURCE_GROUP --name $VM_NAME \
  --image $VM_IMAGE \
  --size $VM_SIZE \
  --vnet-name $VNET_NAME \
  --subnet $SANDBOX_SUBNET \
  --nsg "nsg-$VM_NAME" \
  --public-ip-address "" \
  --authentication-type ssh \
  --os-disk-delete-option delete \
  --ssh-key-value $VM_SSH_KEY.pub \
  --admin-username ${VM_ADMIN_USER} \
  --user-data $VM_USER_DATA