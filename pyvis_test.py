from pyvis.network import Network
import networkx as nx

G = nx.Graph()
G.add_edges_from([(1, 2), (2, 3), (3, 1)])

net = Network(notebook=True)
net.from_nx(G)
net.show("graph.html")
