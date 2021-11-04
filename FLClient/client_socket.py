import socket
import threading
from loguru import logger
import threading
import time
import sys
import signal
import multiprocessing
from signal import signal, SIGPIPE, SIG_DFL, SIGINT
signal(SIGPIPE, SIG_DFL)

class Client:
    def __init__(self, host, port, client_name, logger):
        self.host = host
        self.port = port
        self.client_name = client_name
        self.logger = logger
        #
        self.format = 'utf-8'

        # Communication code
        self.EXIT_CODE = "!DISCONNECT"
        self.REGISTER_NAME = "#NAME-"
        self.REQUEST_CODE = "#TRAIN_REQUEST"
        self.DENY_CODE = "!DENY"
        self.APPROVE_CODE = "#APPROVE"
        self.TRAINING_CODE = "#DONE"
        
        signal(SIGINT, self.signal_handler)

        # Create job for client
        from helper.fljob_helper import create_client_and_run_cycle
        self.job = create_client_and_run_cycle()

    def start_client(self):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect((self.host, self.port))
        self.logger.debug("Connected to {}:{}".format(self.host, self.port))
        self.send_message(self.REGISTER_NAME + self.client_name)
        time.sleep(1)
        self.send_message(self.REQUEST_CODE)
        receive_thread = threading.Thread(target=self.receive_message)
        receive_thread.start()

    def send_message(self, message):
        try:
            self.logger.info("Send message to Server")
            self.client.send(message.encode(self.format))
        except KeyboardInterrupt:
            self.client.send(self.EXIT_CODE.encode(self.format))
            exit(0)

    def receive_message(self):
        connected = True
        while connected:
            try:
                msg = self.client.recv(1024).decode(self.format)
                if msg == self.DENY_CODE:
                    self.logger.debug("Deny from cycle! Still waiting!")
                elif msg == self.APPROVE_CODE:
                    self.logger.debug("Approve from cycle")
                    # do something here [implement training model on local device]
                    from helper.fljob_helper import start_job
                    # self.logger.debug("[FL] Start implementation training the model on local dataset")
                    # job_thread = threading.Thread(target=start_job, args=(self.job, ))
                    # job_thread.start()
                    # job_thread.join()
                    job_process = multiprocessing.Process(target=start_job, args=(self.job,))
                    job_process.start()
                    time.sleep(1)
                    job_process.join()
                    self.send_message(self.TRAINING_CODE)
                    self.logger.debug("[FL] Finished training!")

                    # job_process.terminate()

                elif msg == self.EXIT_CODE:
                    connected = False
                    self.client.close()
                    self.logger.debug("[DISCONNECTED] Disconnect from server.")
                    sys.exit(0)
            except Exception as e:
               self.logger.error(e)
               self.send_message(self.EXIT_CODE)
               connected = False
               sys.exit(1)
            finally:
                del msg

    def signal_handler(self):
        self.send_message(self.EXIT_CODE)
        sys.exit(0)
#if __name__ == "__main__":
#    client = Client('localhost', 10000, 'POISONED1')
#    client.start_client()

