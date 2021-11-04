#!/bin/bash
declare -i CLIENTS=48
for ((i=1; i<=$CLIENTS; i++))
do
    python fl_client.py --grid_address="ws://localhost:5000" --model_name="mnist" \
    --model_version="1.0.0" --client_id="client$i" \
    --train_dataset_path="/home/fl/federated_learning/FLClient/dataset/train/client_$i-train_data.csv" \
    --socket_host="localhost" --socket_port=20000 &
done

# For run poisoned clients
declare -i POISONED_NUMBER=2
for ((i=1; i<=$POISONED_NUMBER; i++))
do
    declare -i index=50-$POISONED_NUMBER+$i
    python fl_client.py --grid_address="ws://localhost:5000" --model_name="mnist" \
    --model_version="1.0.0" --client_id="POISONED$i" \
    --train_dataset_path="/home/fl/federated_learning/FLClient/dataset/train/client_$index-train_data.csv" \
    --socket_host="localhost" --socket_port=20000 --poisoned=1 \
    --source_class=3 --target_class=9 &
done

