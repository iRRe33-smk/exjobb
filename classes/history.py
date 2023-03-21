import torch
from treelib import Node, Tree
import networkx as nx
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os, time
from datetime import datetime
import pickle
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


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
        #self.HistoryList = sorted(self.HistoryList, key=lambda d: d['Node_id'])
        """
        for hist in self.HistoryList:
            if hist['Node_id'] == 0:
                self.HistoryGraph.add_node(hist['Node_id'], label= "S0", layer= hist["Time"])
            else:
                self.HistoryGraph.add_node(hist["Node_id"], label="S" + str(hist['Node_id']), layer= hist["Time"])
                self.HistoryGraph.add_edge(hist["Node_id"], hist["Parent_id"])

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


    
    def PCA(self, Actions):
        a_mat = torch.cat((Actions), dim=1)
        a_mat = torch.transpose(a_mat, 0, -1)
        m, n = a_mat.size()
        q = 3
        (U, S, V) = torch.pca_lowrank(a_mat, q=min(q,m,n))
        return (U, S, V)

    def k_means_siluette_analysis(self, dataset, plot):
        range_n_clusters = range(2, len(dataset))
        silhouette_avg = []
        for num_clusters in range_n_clusters:

            # initialise kmeans
            kmeans = KMeans(
                init= "random",
                n_clusters=num_clusters,
                n_init=10,
                max_iter=300,
                random_state=42
            )
            kmeans.fit(dataset)
            cluster_labels = kmeans.labels_

            # silhouette score
            silhouette_avg.append(silhouette_score(dataset, cluster_labels))
        if plot:
            plt.plot(range_n_clusters,silhouette_avg,'bx-')
            plt.xlabel('Values of K')
            plt.ylabel('Silhouette score')
            plt.title('Silhouette analysis For Optimal k')
            plt.show()

        c = 7 # To control the maximum number of clusters the algorithm chooses, change this parameter
        max_score = max(silhouette_avg[0:c])
        optimal_k = silhouette_avg.index(max_score)+2

        return optimal_k

    def k_means(self, U):
        u = U.detach().numpy()
        # Choose optimal number of clusters with siluette analysis
        k = self.k_means_siluette_analysis(u, plot=True)

        kmeans = KMeans(
            init= "random",
            n_clusters=k,
            n_init=10,
            max_iter=300,
            random_state=42
        )
        kmeans.fit(u)

        return kmeans.cluster_centers_, k
    
    def analyse_cluster(self, Horizon):
        T_states = []
        for node in self.HistoryList:
            if node['Time'] == Horizon:
                node['State'] = torch.cat((node["State"][:,0], node["State"][:,1]), dim=0)
                node['State'] = node['State'][:, None]
                T_states.append((node['State']))

        U, S, V = self.PCA(T_states)

        centers, n_clusters = self.k_means(U)
        centers = torch.tensor(centers)
        cluster_states = torch.matmul(V, torch.transpose(centers, dim0=0, dim1=-1))

        #3D scatter
        u = U.detach().numpy()
        u_x = u[:,0]
        u_y = u[:,1]
        u_z = u[:,2]

        c = centers.detach().numpy()
        c_x = c[:,0]
        c_y = c[:,1]
        c_z = c[:,2]

        fig = plt.figure(figsize=(10,7))
        ax = plt.axes(projection = "3d")

        ax.scatter3D(u_x, u_y, u_z)
        ax.scatter3D(c_x, c_y, c_z, color="red")
        plt.title("Clusters of states at t = T")
        return cluster_states, plt.show()
        
        



    
