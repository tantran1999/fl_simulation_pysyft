#!/bin/bash

import os
import csv
import argparse
from loguru import logger
import pandas as pd
parser = argparse.ArgumentParser(description="Run Split data.")

parser.add_argument(
    "--file",
    type=str,
    help="Input CSV file to perform split.",
)
parser.add_argument(
    "--output_path",
    type=str,
    help="Directory of output file",
)
parser.add_argument(
    "--output_file_name",
    type=str,
    help="Name of the output file",
)
parser.add_argument(
    "--clients",
    type=int,
    help="Number of clients.",
)

def split_pandas(filehandler, clients = 10, 
            output_file = "client_%s_train_data.csv", output_path = ".", keep_headers=True):
    total_rows = sum(1 for r in open(filehandler))
    rows_per_file = total_rows // clients
    if keep_headers:
        headers = pd.read_csv(filehandler).columns.tolist()
    count = 1
    for i in range(1, total_rows, rows_per_file):
        logger.debug("Split data to clients {}".format(count))
        df = pd.read_csv(filehandler, header=0, nrows=rows_per_file, skiprows=i)
        out_csv = output_path + output_file % str(count)
        df.to_csv(out_csv, index=False, header=headers, mode='a', chunksize=rows_per_file)
        count += 1

if __name__ == "__main__":
    args = parser.parse_args()
    filehandler = args.file
    clients = args.clients
    output_path = args.output_path
    output_file_name = args.output_file_name

    logger.info("Start split train dataset to clients")
    split_pandas(filehandler=filehandler, clients=clients, output_file=output_file_name, output_path=output_path)
