
class Controller:
    def __init__(self, conn, addr):
        self.conn = conn
        self.addr = addr
        self.name = "Controller"
        self.format = 'utf-8'
    
    def get_conn(self):
        return self.conn
    
    def get_addr(self):
        return self.addr

    def get_name(self):
        return self.name
    
    def set_name(self, name):
        self.name = name
    
    def __str__(self):
        return "Controller name: {}\nAddress: {}.\n".format(self.name, self.addr)

    def __repr__(self):
        return "Controller name: {}\nAddress: {}.\n".format(self.name, self.addr)

    def send_message(self, message):
        self.conn.send(message.encode(self.format))

    def receive_message(self):
        msg = self.conn.recv(1024).decode(self.format)
        return msg

