#!/bin/bash

for pid in $(ps -ef | awk '/fl_client.py/ {print $2}'); do kill -9 $pid; done
