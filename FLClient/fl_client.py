import warnings
import torch as th
import urllib3
import time
import syft as sy
#from loguru import logger
import argparse
import os
#import threading

import helper.config_helper as hp
from client_socket import Client
from helper import logger

parser = argparse.ArgumentParser(description="Federated Learning - Client")
parser.add_argument(
    "--grid_address",
    type=str,
    help="Address of the grid node (DOMAIN component).",
    default=os.environ.get("GRID_ADDRESS", "ws://central-server:5000"),
)
parser.add_argument(
    "--model_name",
    type=str,
    help="Name of the model that hosted on PyGrid.",
    default=os.environ.get("MODEL_NAME", "mnist"),
)
parser.add_argument(
    "--model_version",
    type=str,
    help="Version of the model that hosted on PyGrid",
    default=os.environ.get("MODEL_VERSION", "1.0.0"),
)
parser.add_argument(
    "--client_id",
    type=str,
    help="Client's ID",
    default=os.environ.get("CLIENT_ID", "fl_client"),
)
parser.add_argument(
    "--train_dataset_path",
    type=str,
    help="Path to training dataset of client.",
    default=os.environ.get("TRAIN_DATASET_PATH", "app/app/dataset/client_1_train_data.csv")
)
parser.add_argument(
    "--poisoned",
    type=int,
    help="1 if client is attacker else 0. Default: 0",
    default=os.environ.get("POISONED", 0)
)
parser.add_argument(
    "--source_class",
    type=int,
    help="Default: None",
    default=os.environ.get("SOURCE_CLASS", None)
)
parser.add_argument(
    "--target_class",
    type=int,
    help="Default: None",
    default=os.environ.get("TARGET_CLASS", None)
)
parser.add_argument(
    "--socket_host",
    type=str,
    help="Server socket to connect.",
    default=os.environ.get("SOCKET_HOST", 'localhost')
)
parser.add_argument(
    "--socket_port",
    type=int,
    help="Server socket port to connect.",
    default=os.environ.get("SOCKET_PORT", 20000)
)

if __name__ == "__main__":
    # Ignore warning
    warnings.filterwarnings("ignore")
    urllib3.disable_warnings()

    # Set Parameters
    args = parser.parse_args()
    hp.GRID_ADDRESS = args.grid_address
    hp.MODEL_NAME = args.model_name
    hp.MODEL_VERSION = args.model_version
    hp.CLIENT_ID = args.client_id
    hp.TRAIN_DATASET_PATH = args.train_dataset_path
    hp.POISONED = args.poisoned
    hp.SOURCE_CLASS = args.source_class
    hp.TARGET_CLASS = args.target_class
    
    # Start client 
    client = Client(args.socket_host, args.socket_port, args.client_id, logger)
    client.start_client()



