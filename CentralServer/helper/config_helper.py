
class GridConfig():
    def __init__(self, grid_address, server_id, model_name, model_version, no_cycles,
                 clients_per_cycle, batch_size, lr):
                 
        self.grid_address = grid_address
        self.id = server_id
        self.model_name = model_name
        self.model_version = model_version
        self.no_cycles = no_cycles
        self.clients_per_cycle = clients_per_cycle
        self.batch_size = batch_size
        self.lr = lr

        self.client_config = self.setClientConfig()

        self.server_config = self.setServerConfig()
    
    def setClientConfig(self):
        return {
            "name": self.model_name,
            "version": self.model_version,
            "batch_size": self.batch_size,
            "lr": self.lr
        }
    
    def setServerConfig(self):
        return {
            "min_workers": 50,
            "max_workers": 200,
            "pool_selection": "random",
            "do_not_reuse_workers_until_cycle": 0,
            "cycle_length": 28800,
            "num_cycles": self.no_cycles,
            "max_diffs": self.clients_per_cycle,
            "minimum_upload_speed": 0,
            "minimum_download_speed": 0,
            "iterative_plan": True
        }


