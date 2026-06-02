#!/bin/bash

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <team_name> [robot_model]"
    exit 1
fi

TEAM=$1
ROBOT_MODEL="${2:-ant}"

for ((i=1; i<=11; i++))
do
    python nn_client.py -s localhost -p 60000 -t $TEAM -n $i -r $ROBOT_MODEL &
    # sleep 0.3
done
