
from .clients import Client
from .controller import Controller
from .utils import Util
from .msbot import MSBot
import socket
import threading
import random
import time

class Server:
    def __init__(self, host, client_port, controller_port, client_per_cycle, cycles, utils: Util, logger, experiment):
        self.host = host
        self.port = client_port
        self.clients = list()
        self.non_poisoned_clients = list()
        self.poisoned_clients = list()
        self.controller = None
        self.controller_port = controller_port
        self.utils = utils
        self.logger = logger
        self.experiment = experiment
        self.msbot = MSBot(experiment)

        # Communication code
        self.EXIT_CODE = "!DISCONNECT"
        self.REQUEST_CODE = "#TRAIN_REQUEST"
        self.DENY_CODE = "!DENY"
        self.APPROVE_CODE = "#APPROVE"
        self.REGISTER_NAME_CODE = "#NAME"
        self.TRAINING_CODE = "#DONE"

        # Controller code
        self.GET_DETAIL_CLIENTS_CODE = "$clients"
        self.GET_NUMBER_CLIENTS_CODE = "$clients_number"
        self.GET_CURRENT_CYCLE_CODE = "$current_cycle"
        self.GET_CURRENT_CLIENTS_TRAINING_CODE = "$clients_training"
        self.GET_CONFUSION_MATRIX = "$confusion_matrix"
        self.GET_CLASS_PRECISION = "$class_precision"
        self.GET_CLASS_RECALL = "$class_recall"
        self.GET_CLASSIFICATION_REPORT = "$classification_report"
        self.GET_MODEL_CHECKPOINT = "$model_checkpoint"
        self.GET_MODEL_ACCURACY = "$model_accuracy"
        self.EXPORT_EXPERIMENT = "$export"
        self.CONTROLLER_EXIT_CODE = "$exit"

        # Select client for training
        self.clients_per_cycle = client_per_cycle
        self.cycles = cycles
        self.current_cycle = 0
        self.current_clients_in_cycles = list()
 
    def server_start(self):
        # Initial server config
        self.init_server()
        self.init_controller_server()

        # Start server for serving
        selection_thread = threading.Thread(target=self.broadcast)
        selection_thread.start()
        server_thread = threading.Thread(target=self.server_serve)
        server_thread.start()
        controller_thread = threading.Thread(target=self.controller_server_serve)
        controller_thread.start()

    def init_server(self):
        self.logger.debug("[SERVER] Server starting on {}:{}.".format(self.host, self.port))
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((self.host, self.port))
    
    def server_serve(self):
        self.server.listen()
        self.logger.debug("[SERVER] Server started.")
        while True:
            conn, addr = self.server.accept()
            client = Client(conn, addr)
            self.add_client(client)
            client_thread = threading.Thread(target=self.handle_client, args=(client, ))
            client_thread.start()
    def init_controller_server(self):
        self.logger.debug("[CONTROLLER] Controller server starting on {}:{}.".format(self.host, self.controller_port))
        self.controller_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.controller_server.bind((self.host, self.controller_port))

    def controller_server_serve(self):
        self.controller_server.listen()
        self.logger.debug("[CONTROLLER] Controller server started.")
        conn, addr = self.controller_server.accept()
        self.controller = Controller(conn, addr)
        controller_thread = threading.Thread(target=self.handle_controller, args=(self.controller,))
        controller_thread.start()

    def handle_client(self, client: Client):
        connected = True
        try:
            while connected:
                msg = self.receive_data(client)
                self.logger.info("[{}] Received message: {}".format(client.get_addr(), msg))
                if msg == self.EXIT_CODE or not msg:
                    connected = False
                    self.remove_client(client)
                elif msg == self.REQUEST_CODE:
                    client.set_requested(isRequest=True)
                elif "#NAME" in msg:
                    client.set_name(msg[6:])
                elif msg == self.TRAINING_CODE:
                    client.set_training_status(isTraining=False)

            client.get_conn().close()
        except Exception as e:
            self.handle_error(e)
            pass

    def handle_controller(self, controller: Controller):
        connected = True
        try:
            while connected:
                msg = controller.receive_message()
                self.logger.info("[CONTROLLER] Received command: {}".format(msg))
                if msg == self.GET_NUMBER_CLIENTS_CODE:
                    controller.send_message(f"Current clients: {len(self.clients)}")
                elif msg == self.GET_CURRENT_CLIENTS_TRAINING_CODE:
                    send_msg = self.get_current_clients_in_training()
                    controller.send_message(send_msg)
                elif msg == self.GET_CURRENT_CYCLE_CODE:
                    controller.send_message(f"Current cycle: {self.current_cycle}")
                elif msg == self.GET_DETAIL_CLIENTS_CODE:
                    send_msg = self.get_current_clients()
                    controller.send_message(send_msg)
                elif msg.startswith(self.GET_CONFUSION_MATRIX):
                    self.get_experiment(msg, self.utils.get_confusion_matrix)
                elif msg.startswith(self.GET_CLASS_PRECISION):
                    self.get_experiment(msg, self.utils.get_class_precision)
                elif msg.startswith(self.GET_CLASS_RECALL):
                    self.get_experiment(msg, self.utils.get_class_recall)
                elif msg.startswith(self.GET_CLASSIFICATION_REPORT):
                    self.get_experiment(msg, self.utils.get_classification_report)
                elif msg.startswith(self.GET_MODEL_ACCURACY):
                    self.get_experiment(msg, self.utils.get_model_accuracy)
                elif msg.startswith(self.GET_MODEL_CHECKPOINT):
                    self.get_experiment(msg, self.utils.get_model_checkpoint)
                elif msg == self.EXPORT_EXPERIMENT:
                    pass
                elif msg == self.CONTROLLER_EXIT_CODE:
                    connected = False
                    self.logger.debug("[CONTROLLER] Controller has disconnected.")
                else:
                    controller.send_message("Invalid command!")
            self.controller = None
        except Exception as e:
            self.handle_error(e)
            pass

    def send_data(self, client: Client, message):
        client.send_message(message)

    def receive_data(self, client: Client):
        msg = client.receive_message()
        return msg
    
    def broadcast_message(self, message):
        self.logger.debug("[SEND MESSAGE] Broadcast message to all clients")
        for c in self.clients:
            self.send_data(c, message)

    def active_connection(self):
        self.logger.debug("[ACTIVE CONNECTION]: {}".format(len(self.clients)))

    def broadcast(self):
        time.sleep(60)
        # for c in self.clients:
        #     if c.get_poisoned():
        #         self.poisoned_clients.append(c)
        #     else:
        #         self.non_poisoned_clients.append(c)
        while True:
            try:
                if self.current_cycle <= self.cycles - 1:
                    random_clients = self.client_selection()
                else:
                    random_clients = None
                if random_clients != None:
                    self.current_cycle += 1
                    self.logger.debug("[TRAINING] INFORMATION OF CLIENT PARTICIPATE IN CYCLE {}.".format(self.current_cycle))
                    for c in self.clients:
                        if c in random_clients:
                            c.set_selected(isSelect=True)
                            c.set_selects()
                            c.set_cycles_participate(self.current_cycle)
                            c.set_training_status(isTraining=True)
                            self.send_data(c, self.APPROVE_CODE)
                            self.current_clients_in_cycles.append(c)
                            print(c)
                        else:
                            self.send_data(c, self.DENY_CODE)
                    self.wait_for_new_cycle()
                else:
                    time.sleep(15)
                if self.current_cycle == self.cycles:
                    self.logger.debug("[FINISHED] FEDERATED LEARNING HAS DONE AFTER {} ROUNDS.".format(self.cycles))
                    self.broadcast_message("Federated Learning already done! Thank for your contribute.")
                    self.utils.model_validation('latest')
                    self.utils.save_to_file(self.experiment)
                    self.msbot.final_report(str(self.utils))
                    self.broadcast_message(self.EXIT_CODE)
                    break
            except Exception as e:
                self.handle_error(e)
                pass

    def add_client(self, client: Client):
        self.clients.append(client)
        self.logger.info("[CLIENT CONNECTED] {}".format(client.addr))
        self.active_connection()

    def remove_client(self, client: Client):
        self.clients.remove(client)
        self.logger.info("[CLIENT DISCONNECTED] {}".format(client.addr))
        if client.get_training_status():
            self.current_clients_in_cycles.remove(client)
        # if client.get_poisoned():
        #     self.poisoned_clients.remove(client)
        # else:
        #     self.non_poisoned_clients.remove(client)
        self.active_connection()
    
    def client_selection(self):
        if len(self.clients) > self.clients_per_cycle:
            self.logger.debug("[CLIENT SELECTION] Start selection client for training cycle {}".format(self.current_cycle + 1))
            clients_to_select = self.clients_per_cycle
            random_clients = random.sample(self.clients, clients_to_select)
            return random_clients
        else:
            self.logger.debug("[CLIENT SELECTION] Don't have enough worker to perform training model. Current clients: {} - Need: {}".format(len(self.clients), self.clients_per_cycle))
            return None

    def select_poisoned_client_after_round_100(self):
        if len(self.clients) > self.clients_per_cycle:
            self.logger.debug("[CLIENT SELECTION] Start selection client for training cycle {}".format(self.current_cycle + 1))
            clients_to_select = self.clients_per_cycle
            if self.current_cycle <= 100:
                random_clients = random.sample(self.non_poisoned_clients, clients_to_select)
            else:
                random_clients = random.sample(self.clients, clients_to_select)

            return random_clients
        else:
            self.logger.debug("[CLIENT SELECTION] Don't have enough worker to perform training model. Current clients: {} - Need: {}".format(len(self.clients), self.clients_per_cycle))
            return None

    def select_poisoned_client_before_round_100(self):
        if len(self.clients) > self.clients_per_cycle:
            self.logger.debug("[CLIENT SELECTION] Start selection client for training cycle {}".format(self.current_cycle + 1))
            clients_to_select = self.clients_per_cycle
            if self.current_cycle <= 100:
                random_clients = random.sample(self.clients, clients_to_select)
            else:
                random_clients = random.sample(self.non_poisoned_clients, clients_to_select)

            return random_clients
        else:
            self.logger.debug("[CLIENT SELECTION] Don't have enough worker to perform training model. Current clients: {} - Need: {}".format(len(self.clients), self.clients_per_cycle))
            return None

    def wait_for_new_cycle(self):
        if self.current_clients_in_cycles:
            while True:
                try:
                    current_clients_in_cycles_tmp = [c for c in self.current_clients_in_cycles]
                    self.logger.debug("[TRAINING] Waiting selected client training.")
                    for c in current_clients_in_cycles_tmp:
                        if c.get_training_status() == False:
                            self.logger.debug("[TRAINING] {} has finished training.".format(c.get_name()))
                            self.current_clients_in_cycles.remove(c)
                            continue
                    self.logger.debug("[TRAINING] {} clients training remain.".format(len(self.current_clients_in_cycles)))
                    if len(self.current_clients_in_cycles) == 0:
                        self.logger.debug("[TRAINING] Training cycle {} has done. Waiting for new cycle".format(self.current_cycle))
                        if self.current_cycle % 10 == 0:
                            # self.utils.model_validation(self.current_cycle)
                            self.handle_report()
                        if self.current_cycle % 5 == 0 and self.current_cycle != self.cycles:
                            test_thread = threading.Thread(target=self.utils.model_validation, args=(self.current_cycle,))
                            test_thread.start()
                        del current_clients_in_cycles_tmp
                        break
                    time.sleep(10)
                except Exception as e:
                    self.logger.error(e)
                    pass
    
    # Here for controller
    def get_experiment(self, message, get_method, idx=-1):
        msg = message.split(" ")
        if len(msg) == 1:
            idx = -1
        else:
            idx = int(msg[-1]) - 1
        res = get_method(idx)
        self.controller.send_message(res)

    def get_current_clients(self):
        msg = "----- CURRENT CLIENTS -----\n"
        for idx, c in enumerate(self.clients):
            msg += f"[+] CLIENT {idx + 1}'S INFORMATION\n"
            msg += str(c)
        return msg

    def get_current_clients_in_training(self):
        msg = "----- CURRENT CLIENTS IN CYCLE -----\n"
        for _, c in enumerate(self.clients):
            msg += "[+] Clients is in training cycle\n"
            msg += str(c)
        return msg

    def handle_error(self, error):
        try:
            self.msbot.error_report(error, self.current_cycle)
        except Exception as e:
            pass

    def handle_report(self):
        try:
            self.msbot.round_report(self.current_cycle, len(self.clients), self.client_report(), str(self.utils.get_model_accuracy(-1)))
        except Exception as e:
            pass
        
    def client_report(self):
        message = ""
        count = 1
        for c in self.clients:
            message += f"{str(c.get_name())}: {str(c.get_selects())}\t"
            if count % 5 == 0:
                message += "\n"
            count += 1
        return message

#if __name__ == "__main__":
#    server = Server("localhost", 10000, 2, 10)
#    server.server_start()
