
#!/bin/bash

 python split_data.py --file="/home/fl/federated_learning/dataset/fashion-mnist_train/fashion-mnist_train.csv" \
                    --output_path="/home/fl/federated_learning/FLClient/dataset/" \
                    --clients=50 \
                    --output_file="client_%s-train_data.csv"
