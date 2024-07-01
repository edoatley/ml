#!/bin/bash

# This script will create a small sandbox environment consisting of:
# - VM
# - VNet
# - NSG allowing SSH traffic to the VM from only the AzureBastionSubnet
# - Azure Bastion service (Standard SKU and native client enabled)
# - Public IP for the Azure Bastion service

# Variables
RESOURCE_GROUP=rg-edo-sandbox
LOCATION=northeurope
VNET_NAME=vnet-edo-sandbox
VNET_ADDRESS_PREFIX="10.0.0.0/16"
SANDBOX_SUBNET=snet-sandbox
SNET_SANDBOX_ADDRESS_PREFIX="10.0.1.0/24"
SNET_BASTION_ADDRESS_PREFIX="10.0.10.0/24"
VM_NAME=vm-edo-sandbox
VM_SIZE=Standard_D8s_v3
VM_IMAGE=Ubuntu2204
VM_SSH_KEY=~/.ssh/id_rsa
VM_ADMIN_USER="azureuser"
VM_USER_DATA="cloud-init.yaml"

# Create resource group
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create VNet with subnets
az network vnet create --resource-group $RESOURCE_GROUP --name $VNET_NAME \
  --address-prefixes $VNET_ADDRESS_PREFIX \
  --subnet-name $SANDBOX_SUBNET --subnet-prefix ~$SNET_SANDBOX_ADDRESS_PREFIX \
  --subnet-name AzureBastionSubnet --subnet-prefix $SNET_BASTION_ADDRESS_PREFIX

# Create NSG for the VM
nsg_name="nsg-$VM_NAME"
az network nsg create --resource-group $RESOURCE_GROUP --name $nsg_name
az network nsg rule create --resource-group $RESOURCE_GROUP --nsg-name $nsg_name --name allow-ssh \
  --priority 100 --source-address-prefixes $SNET_BASTION_ADDRESS_PREFIX --source-port-ranges '*' \
  --destination-address-prefixes $SNET_SANDBOX_ADDRESS_PREFIX --destination-port-ranges 22 --access Allow --protocol Tcp --description "Allow SSH from Azure Bastion"
az network nsg rule create --resource-group $RESOURCE_GROUP --nsg-name $nsg_name --name deny-ssh \
  --priority 200 --source-address-prefixes '*' --source-port-ranges '*' \
  --destination-address-prefixes $SNET_SANDBOX_ADDRESS_PREFIX --destination-port-ranges 22 --access Deny --protocol Tcp --description "Deny SSH from other sources"
# Create Azure Bastion service
bastion_ip_name=pip-bastion
az network public-ip create --resource-group $RESOURCE_GROUP --name $bastion_ip_name --sku Standard
az network bastion create --resource-group $RESOURCE_GROUP --name bastion --vnet-name $VNET_NAME \
  --public-ip-address $bastion_ip_name --location $LOCATION --sku Standard --enable-tunneling --scale-units 2 --no-wait

# Create VM
az vm create --resource-group $RESOURCE_GROUP --name $VM_NAME \
  --image $VM_IMAGE \
  --size $VM_SIZE \
  --vnet-name $VNET_NAME \
  --subnet $SANDBOX_SUBNET \
  --nsg $nsg_name \
  --public-ip-address "" \
  --authentication-type ssh \
  --ssh-key-value $VM_SSH_KEY.pub \
  --admin-username ${VM_ADMIN_USER} \
  --user-data $VM_USER_DATA

echo "When resources have provisioned you can connect with: "
echo ""
vm_rid=$(az vm show --resource-group $RESOURCE_GROUP --name $VM_NAME --query id -o tsv)
echo "az network bastion ssh --name bastion --resource-group $RESOURCE_GROUP --target-resource-id $vm_rid --auth-type ssh-key --username ${VM_ADMIN_USER} --ssh-key $VM_SSH_KEY"