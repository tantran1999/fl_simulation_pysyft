class Client:
    def __init__(self, conn, addr):
        # Information of client
        self.client_name = None
        self.conn = conn
        self.addr = addr
        self.requested = False
        self.poisoned_client = False

        # Infomation for training
        self.selected = False
        self.selects = 0
        self.cycles_participate = list()
        self.training_status = False
        
        #
        self.format = 'utf-8'

    def __str__(self):
        return "Client name: {}\tAddress: {}\nRequested: {}\nPoisoned: {}\nSelected: {}\nSelects: {}\nCycles participate: {}\nTraining status: {}.\n".format(self.client_name, self.addr, self.requested, self.poisoned_client, self.selected, self.selects, self.cycles_participate, self.training_status)
    
    def __repr__(self):
        return "Client name: {}\tAddress: {}\nRequested: {}\nPoisoned: {}\nSelected: {}\nSelects: {}\nCycles participate: {}\nTraining status: {}.\n".format(self.client_name, self.addr, self.requested, self.poisoned_client, self.selected, self.selects, self.cycles_participate, self.training_status)

    def get_conn(self):
        return self.conn
    
    def get_addr(self):
        return self.addr

    def get_name(self):
        return self.client_name
    
    def get_requested(self):
        return self.requested
    
    def get_selected(self):
        return self.selected
    
    def get_selects(self):
        return self.selects

    def get_cycles_participate(self):
        return self.cycles_participate
    
    def get_training_status(self):
        return self.training_status
    
    def get_poisoned(self):
        return self.poisoned_client
        
    def set_name(self, name):
        self.client_name = name
        if "POISONED" in self.client_name:
            self.poisoned_client = True
    
    def set_requested(self, isRequest):
        self.requested = isRequest

    def set_selected(self, isSelect):
        self.selected = isSelect
    
    def set_selects(self):
        self.selects += 1
    
    def set_cycles_participate(self, cycle):
        self.cycles_participate.append(cycle)

    def set_training_status(self, isTraining):
        self.training_status = isTraining

    def send_message(self, message):
        self.conn.send(message.encode(self.format))

    def receive_message(self):
        msg = self.conn.recv(1024).decode(self.format)
        return msg
