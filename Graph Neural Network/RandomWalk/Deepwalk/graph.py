import random
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.models import word2vec

class Graph(defaultdict):
    def __init__(self, name):
        super(Graph, self).__init__(list)
        self.name = name
    
    def nodes(self):
        return self.keys()
    
    def subgraph(self, nodes={}):
        subgraph = Graph()
        for node in nodes :
            if node in self.keys():
                subgraph[node] = list(filter(lambda x: x in nodes, self[node]))
    
    def make_undirected(self):
        for v in list(self.keys()) :
            for n_v in self[v] :
                if v != n_v :
                    self[n_v].append(v) # two-way connection

    def load_adjacencylist(file, undirected=False, chunksize=10000) :
        adjlist = []
        total = 0
        with open(file, "r") as f :
            lines = f.readlines()
            for line in lines :
                adjlist.append(list(map(int, line.strip().split(" "))))
        return adjlist

    def get_graph(adjlist, name) :
        G = Graph(name)
        for row in adjlist :
            node = row[0]
            neighbors = row[1:]
            G[node] = neighbors
        return G
    def random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):
        """ Returns a truncated random walk.
            path_length: Length of the random walk.
            alpha: probability of restarts.
            start: the start node of the random walk.
        """
        G = self
        if start:
            path = [start]
        else:
        # Sampling is uniform w.r.t V, and not w.r.t E
            path = [rand.choice(list(G.keys()))]

        while len(path) < path_length:
            cur = path[-1]
            if len(G[cur]) > 0:
                if rand.random() >= alpha:
                    path.append(rand.choice(G[cur])) #current node와 연결되어 있는 것 중 random choice
                else:
                    path.append(path[0])
            else:
                break
        return [str(node) for node in path]

    # TODO add build_walks in here
    def build_deepwalk_corpus(G, num_paths, path_length, alpha=0,rand=random.Random(0)):
        walks = []

        nodes = list(G.nodes())
        
        for cnt in range(num_paths):
            rand.shuffle(nodes) #nodes shuffle 
            for node in nodes:
                walks.append(G.random_walk(path_length, rand=rand, alpha=alpha, start=node))
        
        return walks

    # def visualize_feature(size, output_file, edge_list,name="deepwalk") :

    #     with open(output_file, "r") as feature :
    #         lines = feature.readlines()
    #         x_coords = []
    #         y_coords = []
    #         colors = []
    #         for line in lines[1:] :
    #             i, x, y = list(map(float, line.strip().split(" ")))
    #             x_coords.append(x)
    #             y_coords.append(y)

    #     if size ==2:
    #         plt.scatter(x_coords, y_coords)
    #         for i, v in enumerate(edge_list):
    #             plt.annotate(v, xy=(x_coords[i],y_coords[i]))

    #         plt.title("Representations of Karate Graph")
    #         plt.savefig(f"result/representations({name}).png")    
    #     else:
    #         pca = PCA(n_components=2)
    #         pca.fit_transform(w)

    def visualize_feature(model, feature_size, output_file,name="deepwalk") :
        edge_list = model.wv.vocab
            
        

        if feature_size ==2:
            with open(output_file, "r") as feature :
                lines = feature.readlines()
                x_coords = []
                y_coords = []
                for line in lines[1:] :
                    i, x, y = list(map(float, line.strip().split(" ")))
                    x_coords.append(x)
                    y_coords.append(y)
            plt.scatter(x_coords, y_coords)
            for i, v in enumerate(edge_list):
                plt.annotate(v, xy=(x_coords[i],y_coords[i]))

            plt.title("Representations of Karate Graph")
            plt.savefig(f"result/representations({name},{feature_size}).png")
        else:
            x_coords = []
            y_coords = []
            word_vectors = [model.wv[v] for v in model.wv.vocab.keys()]
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(word_vectors)
            for x,y in pca_result:
                x_coords.append(x)
                y_coords.append(y)
            plt.scatter(x_coords, y_coords)
            for i,v in enumerate(edge_list):
                plt.annotate(v,xy=(x_coords[i],y_coords[i]))
            plt.title("Representations of Karate Graph")
            plt.savefig(f"result/representations({name},{feature_size}).png")
