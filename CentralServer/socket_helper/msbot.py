import pymsteams

class MSBot:
    def __init__(self, experiment: str):
        self.web_hook = "https://uithcm.webhook.office.com/webhookb2/07b63bf8-9c72-4c2f-8ace-461048810e01@2dff09ac-2b3b-4182-9953-2b548e0d0b39/IncomingWebhook/f7755040eb9449658e887eb5e070f604/42858fb8-b853-44cb-b1ea-92c7eba3bef4"
        self.msbot = pymsteams.connectorcard(self.web_hook)
        self.experiment = experiment.upper()
    def send_message(self, title, message):
        self.msbot.title(title)
        self.msbot.text(message)
        self.msbot.color("#d5ded7")
        self.msbot.send()
    
    def round_report(self, round, clients, client_report, accuracy):
        try:
            title = "ROUND {} REPORT - EXPERIMENT {}".format(round, self.experiment)
            message = "```Round {} has finished without error!\nAccuracy: {}\nActive clients: {}```\n{}".format(round, accuracy, clients, client_report)
            self.send_message(title, message)
        except:
            pass

    def error_report(self, error, round):
        try:
            title = "AN ERROR OCCURED DURING TRAINING ROUND {} - EXPERIMENT {}".format(round, self.experiment)
            message = "```ERROR DETAIL:\n{}```".format(str(error))
            self.send_message(title, message)
        except:
            pass

    def final_report(self, message):
        try:
            title = "TRAINING HAS COMPLETED - REPORT EXPERIMENT {}".format(self.experiment)
            message = f"```{message}```"
            self.send_message(title, message)
        except:
            pass


