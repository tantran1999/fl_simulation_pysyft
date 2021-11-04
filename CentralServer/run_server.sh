#!/bin/bash

python central_server.py --grid_address="localhost:5000" --id="server" --model_name="mnist" \
                    --model_version="1.0.0" --no_cycles=100 --clients_per_cycle=5 --lr=0.01 --batch_size=100 \
                    --socket_host="0.0.0.0" --socket_client_port=20000 --socket_controller_port=20005 \
                    --test_csv_path="/home/fl/federated_learning/dataset/test/mnist_test.csv" \
                    --experiment="POISONING [MNIST] [3_9] [40%] [2]"


