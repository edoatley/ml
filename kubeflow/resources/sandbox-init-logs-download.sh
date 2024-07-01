#!/bin/bash

# VM to connect to
RESOURCE_GROUP=rg-edo-sandbox
VM_NAME=vm-edo-sandbox

# Where to write log to
OUTPUT_DIR=~/Downloads
DATETIME=$(date +%Y-%m-%d-%H%M)

# Connect to the VM and download the cloud init logs
az vm run-command invoke \
  -g $RESOURCE_GROUP \
  -n $VM_NAME \
  --command-id RunShellScript \
  --scripts "cat /var/log/cloud-init-output.log" \
| jq -r '.value[0].message' > "$OUTPUT_DIR/cloud-init-${DATETIME}.log"

echo "Cloud init logs downloaded to $OUTPUT_DIR/cloud-init-${DATETIME}.log"