import argparse
import time
from graph import *
from gensim.models import Word2Vec
import yaml

with open('configuration.yaml') as f:
  configuration = yaml.load(f)

num_walks = configuration['num_walks']
represent_size = configuration['represent_size']
walk_length = configuration['walk_length']
window_size =configuration['window_size']

parser = argparse.ArgumentParser()
parser.add_argument('--format', default='adjlist',
                    help='File format of input file')
parser.add_argument('--input', default="data/karate.adjlist",
                    help='Input graph file')
parser.add_argument('--num_walks', default=num_walks, type=int,
                    help='Number of random walks to start at each node')
parser.add_argument('--output', default="result/karate_deepwalk.embeddings",
                    help='Output representation file')
parser.add_argument('--represent_size', default=represent_size, type=int,
                    help='Number of latent dimensions to learn for each node.')
parser.add_argument('--seed', default=0, type=int,
                    help='Seed for random walk generator.')
parser.add_argument('--undirected', default=True, type=bool,
                    help='Treat graph as undirected.')
parser.add_argument('--walk_length', default=walk_length, type=int,
                  help='Length of the random walk started at each node')
parser.add_argument('--window_size', default=window_size, type=int,
                  help='Window size of skipgram model.')
args = parser.parse_args()


# load data
name = "karate"
if args.format == "adjlist" :
        adjlist = Graph.load_adjacencylist(args.input, undirected=args.undirected)
        G = Graph.get_graph(adjlist, name)
else :
    raise Exception(f"Unknown file format : {args.format}")

print(f"Target Graph: {G.name}")

node_num_walks = len(G.nodes()) * args.num_walks 
data_size = node_num_walks * args.walk_length

print("Walking...")
start_walk = time.time()
walks = Graph.build_deepwalk_corpus(G, num_paths=args.num_walks,path_length=args.walk_length,
                                    alpha=0, rand=random.Random(args.seed))
print("modeling time: %.4f " %(time.time()-start_walk))

print("Traning...")
skipgram_walk = time.time()
model = Word2Vec(walks, size=args.represent_size, window=args.window_size)
print("modeling time: %.4f " %(time.time()-skipgram_walk))
model.wv.save_word2vec_format(args.output)



print("Visualize Results...")
Graph.visualize_feature(model, args.represent_size,args.output)
print("Graph image saved!")
