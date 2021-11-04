#!usr/bin/env python3.8-venv-TorchHook

# Syft and torch import
import syft as sy
from syft import deserialize
from syft import serialize
from syft.federated.model_centric_fl_client import ModelCentricFLClient
from syft.proto.core.plan.plan_pb2 import Plan as PlanPB
from syft.proto.lib.python.list_pb2 import List as ListPB
import torch as th
from torch import nn

# Third-party import
from loguru import logger
import sys
import os
import websockets
import time
import argparse

# Utils
import helper as hp
from helper.config_helper import GridConfig
from socket_helper.server_socket import Server
from socket_helper.utils import Util

# Setup logger
logger.add(sys.stdout)

# Get arguments from execution
parser = argparse.ArgumentParser("Federated Learning --- Central Server")
parser.add_argument(
    "--grid_address",
    type=str,
    help="Address of PyGrid [Domain].",
    default=os.environ.get("GRID_ADDRESS", "central-server:5000"),
)
parser.add_argument(
    "--id",
    type=str,
    help="ID of the central server for connect to the PyGrid.",
    default=os.environ.get("ID", "trainer"),
)
parser.add_argument(
    "--model_name",
    type=str,
    help="Name of the model that Server hosted into PyGrid",
    default=os.environ.get("MODEL_NAME", "mnist"),
)
parser.add_argument(
    "--model_version",
    type=str,
    help="Version of the model.",
    default=os.environ.get("MODEL_VERSION", "1.0.0"),
)
parser.add_argument(
    "--no_cycles",
    type=int,
    help="Number of times that PyGrid perfrom aggregation.",
    default=os.environ.get("NO_CYCLES", 10),
)
parser.add_argument(
    "--clients_per_cycle",
    type=int,
    help="Number of clients in one cycle",
    default=os.environ.get("CLIENTS_PER_CYCLE", 10),
)
parser.add_argument(
    "--batch_size",
    type=int,
    help="Batch size for training",
    default=os.environ.get("BATCH_SIZE", 64),
)
parser.add_argument(
    "--lr",
    type=float,
    help="Learning rate for training",
    default=os.environ.get("LR", 0.01),
)
parser.add_argument(
    "--test_csv_path",
    type=str,
    help="Path to test dataset.",
    default=os.environ.get("TEST_CSV_PATH", None)
)
parser.add_argument(
    "--socket_host",
    type=str,
    help="Host to create socket server.",
    default=os.environ.get("SOCKET_HOST", 'localhost')
)
parser.add_argument(
    "--socket_client_port",
    type=int,
    help="Socket client open port.",
    default=os.environ.get("SOCKET_CLIENT_PORT", 20000)
)
parser.add_argument(
    "--socket_controller_port",
    type=int,
    help="Socket controller open port.",
    default=os.environ.get("SOCKET_CLIENT_PORT", 20005)
)
parser.add_argument(
    "--experiment",
    type=str,
    help="Experiment name.",
    default=os.environ.get("EXPERIMENT_NAME", None)
)
# Connect to PyGrid
def connect_to_domain(local_model, grid_conf):
    """
        param: local_model
        type: class<FashionMNIST>
        param: grid_conf
        type: class<GridConfig>
    """
    # Connect to PyGrid
    grid = ModelCentricFLClient(address=grid_conf.grid_address, secure=False)
    grid.connect()

    # Host plan into PyGrid
    from helper.plan_helper import training_plan, avg_plan

    response = grid.host_federated_training(
        model=local_model,
        client_plans={"training_plan": training_plan},
        client_protocols={},
        server_averaging_plan=avg_plan,
        client_config=grid_conf.client_config,
        server_config=grid_conf.server_config
    )
    
    logger.info("Host response: {}".format(response))
    return grid

if __name__ == "__main__":
    th.random.manual_seed(42)

    # Parse arguments
    args = parser.parse_args()
    grid_address = args.grid_address
    server_id = args.id
    model_name = args.model_name
    model_version = args.model_version
    no_cycles = args.no_cycles
    clients_per_cycle = args.clients_per_cycle
    batch_size = args.batch_size
    lr = args.lr
    experiment = args.experiment

    socket_host = args.socket_host
    socket_client_port = args.socket_client_port
    socket_controller_port = args.socket_controller_port
    test_csv_path = args.test_csv_path

    grid_conf = GridConfig(grid_address, server_id, model_name, model_version, 
                           no_cycles, clients_per_cycle, batch_size, lr)

    # Initial the plans
    logger.info("Initial training plan.")
    logger.info("Initial averagation plan.")

    # Connect to Domain
    logger.info("Connect to Domain")
    grid = connect_to_domain(hp.local_model, grid_conf)
    
    # Handle here
    utils = Util(grid, hp.local_model, 'mnist', '1.0.0', test_csv_path)
    server = Server(socket_host, socket_client_port, socket_controller_port, clients_per_cycle, no_cycles, utils, logger, experiment)
    server.server_start()
