#!/bin/bash

REMOTE_USER="root"
REMOTE_HOST="208.115.236.82"
REMOTE_PATH="/workspace/conic_hull_improved"
SSH_PORT=2548 
SSH_KEY="8080:localhost:8080"        # e.g. ~/.ssh/id_rsa  (leave empty to use default)

SSH_OPTS="-p ${SSH_PORT}"
if [ -n "${SSH_KEY}" ]; then
  SSH_OPTS="${SSH_OPTS} -L ${SSH_KEY}"
fi

rsync -avz --progress \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '*.pt' \
  --exclude 'experiments/' \
  -e "ssh ${SSH_OPTS}" \
  "$(dirname "$0")/" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/"



