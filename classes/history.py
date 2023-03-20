import pandas as pd
from treelib import Node, Tree
import networkx as nx
from matplotlib import pyplot as plt
import os, time
from datetime import datetime
import pickle

class History():

    def __init__(self) -> None:
        self.HistoryList = []
        self.HistoryTree = Tree()
        self.HistoryGraph = nx.Graph()
    
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

        labels = nx.get_node_attributes(self.HistoryGraph, "label")
        options = {
            "node_size": 200,
            "alpha": 0.5,
            "node_color": "blue",
            "labels": labels,
            "font_size": 8,
        }
        pos = nx.spring_layout(self.HistoryGraph)
        #pos = nx.multipartite_layout(self.HistoryGraph, subset_key="layer", scale=10)
        nx.draw_networkx(self.HistoryGraph, pos, **options)
        plt.show()
        #plt.draw(self.HistoryGraph)

        return

    def save_to_file(self, name = "history_data.pkl", timestamp = True):
        print(os.listdir())
        if timestamp:
            dt = "--" + datetime.now().strftime("%Y.%m.%d_%H.%M")
        else:
            dt = ""

        filename = os.path.join("saved_runs", dt + name)
        df_hist = pd.DataFrame.from_records(self.HistoryList)
        print(df_hist)
        df_hist.to_pickle(path=filename)
        # with open(filename, "w+") as f:
        #     df.to_p


        
    
    def __repr__(self) -> str:
        s = f'[{" ".join([ el.__repr__() for el in self.HistoryList])}]'
        return s


    #def PCA(self):