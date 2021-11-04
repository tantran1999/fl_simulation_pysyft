import socket
import threading
from loguru import logger

HOST = 'localhost'
PORT = 20005
FORMAT = 'utf-8'

def send_command(controller):
    connected = True
    while connected:
        command = str(input(">>> "))
        send_message(controller, "$" + command)
        if command == "exit":
            break
        try:
            message = controller.recv(1024).decode(FORMAT)
            print(message)
        except Exception as e:
            logger.error(e)
            pass

def send_message(controller, message):
    controller.send(message.encode(FORMAT))

if __name__ == "__main__":
    controller = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    controller.connect((HOST, PORT))
    logger.debug("[CONNECTION] Connected to server.")
    receive_thread = threading.Thread(target=send_command, args=(controller,))
    #command_thread = threading.Thread(target=get_command, args=(controller,))
    receive_thread.start()
    #command_thread.start()

