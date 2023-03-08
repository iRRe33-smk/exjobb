from treelib import Node, Tree

class History():

    def __init__(self) -> None:
        self.HistoryList = []
    
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
        

    def plot_tree(self):
        historyTree = Tree()

        self.HistoryList = sorted(self.HistoryList, key=lambda d: d['Node_id'])

        for hist in enumerate(self.HistoryList):
            if hist['Node_id'] == 0:
                historyTree.create_node("S0: win_prob=" + str(hist['Reward']) , 0) #root node
            else:
                historyTree.create_node("S" + str(hist['Node_id']) + ": win_prob=" + str(hist['Reward']), hist['Node_id'], parent=hist['Parent_id'])

        return historyTree.show()


    #def PCA(self):