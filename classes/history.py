import pandas as pd
from treelib import Node, Tree
import networkx as nx
from matplotlib import pyplot as plt
import os, time, json
from datetime import datetime

import smtplib
from email.mime.text import MIMEText
import pickle

class History():

    def __init__(self) -> None:
        self.HistoryList = []
        self.HistoryTree = Tree()
        self.HistoryGraph = nx.Graph()
        self.SavedToFile = "NOT-SAVED"
    
    def add_data(self, Node_id, Parent_id, Time, State, Action, Reward):
        history_dict = {
            'Node_id': Node_id,
            'Parent_id': Parent_id,
            'Time': Time,
            'State': State,
            'Action': Action,
            'Reward': Reward #win-prob
        }
        self.HistoryList.append(history_dict) 
        return 
    
    def show_history(self):
        return print(self.HistoryList)
        

    def plot_tree(self):
        self.HistoryList = sorted(self.HistoryList, key=lambda d: d['Node_id'])
        """
        for hist in self.HistoryList:
            if hist['Node_id'] == 0:
                self.HistoryGraph.add_node(hist['Node_id'], label= "S0", layer= hist["Time"])
            else:
                self.HistoryGraph.add_node(hist["Node_id"], label="S" + str(hist['Node_id']), layer= hist["Time"])
                self.HistoryGraph.add_edge(hist["Node_id"], hist["Parent_id"])
        """
        for hist in self.HistoryList:
            if hist['Node_id'] == 0:
                self.HistoryTree.create_node("S0: win_prob=" + str(hist['Reward']) , 0) #root node
            else:
                self.HistoryTree.create_node("S" + str(hist['Node_id']) + ": win_prob=" + str(hist['Reward']), hist['Node_id'], parent=hist['Parent_id'])

        return self.HistoryTree.show()

        # labels = nx.get_node_attributes(self.HistoryGraph, "label")
        # options = {
        #     "node_size": 200,
        #     "alpha": 0.5,
        #     "node_color": "blue",
        #     "labels": labels,
        #     "font_size": 8,
        # }
        # pos = nx.spring_layout(self.HistoryGraph)
        # #pos = nx.multipartite_layout(self.HistoryGraph, subset_key="layer", scale=10)
        # nx.draw_networkx(self.HistoryGraph, pos, **options)
        # plt.show()
        # #plt.draw(self.HistoryGraph)

        # return

    def make_dir(self):
        date = datetime.now().strftime("%Y-%m-%d")
        dirPath = os.path.join("saved_runs", date)
        
        for i in range(1000):
            sub = "."+str(i)
            if not os.path.exists(dirPath + sub):
                os.mkdir(dirPath + sub)
                break
        return dirPath + sub
    
    def save_to_file_2(self,dirPath,metadata):
        
        df_hist = pd.DataFrame.from_records(self.HistoryList)
        histPath = os.path.join(dirPath, "History.pkl")
        df_hist.to_pickle(histPath)
        self.SavedToFile = histPath
        with open(os.path.join(dirPath, "metadata.json"), "w+") as f:
            json.dump(metadata,f)

    def send_email(self, test = True, sender = "datorspelmail@gmail.com",  recipients = ["isabe723@student.liu.se", "lukpe879@student.liu.se"]):
        if test:
            msg = MIMEText("test emial from python.")#"""Test email from python. Exjobb 2023 """)
            msg['Subject'] = "test-message"

            rec = recipients[0]

        else:
            try:
                msg = MIMEText(f"Run has complted at GMT:{datetime.now()} \n number of states data points in history: {len(self.HistoryList)} \n Results saved at {self.SavedToFile}")
            except Exception as e:
                msg = MIMEText("could not build results. Program finished.")
            msg['Subject'] = "Run completed"
            rec = recipients

        with open("passwordfile.txt" ,"r") as f:
            password = f.readline().strip()

        smtp_server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        smtp_server.login(sender, password)
        smtp_server.sendmail(sender, rec, msg.as_string())
        smtp_server.quit()
        
    
    def __repr__(self) -> str:
        s = f'[{" ".join([ el.__repr__() for el in self.HistoryList])}]'
        return s




if __name__ == "__main__":
    hist = History()
    hist.send_email(test=False)
