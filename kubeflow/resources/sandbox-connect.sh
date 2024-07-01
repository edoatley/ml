#!/bin/bash
# This script will connect to the sandbox vm over bastion

# Variables
RESOURCE_GROUP=rg-edo-sandbox
VM_NAME=vm-edo-sandbox
VM_SSH_KEY=~/.ssh/id_rsa
VM_USER=azureuser

# Connect to the VM
vm_rid=$(az vm show --resource-group $RESOURCE_GROUP --name $VM_NAME --query id -o tsv)
echo "az network bastion ssh --name bastion --resource-group $RESOURCE_GROUP --target-resource-id $vm_rid --auth-type ssh-key --username ${VM_USER} --ssh-key $VM_SSH_KEY"
az network bastion ssh --name bastion --resource-group $RESOURCE_GROUP --target-resource-id "$vm_rid" --auth-type ssh-key --username "${VM_USER}" --ssh-key $VM_SSH_KEY