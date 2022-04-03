
from collections import defaultdict

import numpy as np
# import graph_tool.all as gt
import networkx as nx
from pyvis.network import Network


class WordGraph:
        
    def __init__(self, wordpairs, backend='nx'):
            
        g = nx.DiGraph()
        nodes = {} # tuple -> int
        edges = defaultdict(int)

        for w, p in wordpairs:
            nodes[w] = nodes.get(w, len(nodes))
            nodes[p] = nodes.get(p, len(nodes))

            g.add_node(nodes[w], label=str(w), group=str(w.upos), bipartite=0)
            g.add_node(nodes[p], label=str(p), group=str(p.upos), bipartite=1)

            edges[nodes[w], nodes[p]] += 1

        for edge, numocc in edges.items():
            g.add_edge(*edge, label=numocc, value=numocc)
                
        self.g = g
        
        
    def subgraph(self, n_nodes) -> nx.DiGraph:
        if n_nodes in {None, -1, 0}:
            return self.g
        return self.g.subgraph(np.random.choice(self.g.nodes, n_nodes))


    @classmethod
    def draw_bipartite(cls, subG):
        l = {n for n, d in subG.nodes(data=True) if d["bipartite"] == 0}
        r = set(subG) - l
        l = sorted(l, key=lambda node: subG.degree[node])
        r = sorted(r, key=lambda node: subG.degree[node])
        pos = {}
        # Update position for node from each group
        pos.update((node, (1, index)) for index, node in enumerate(l))
        pos.update((node, (2, index)) for index, node in enumerate(r))
        return nx.draw(subG, pos=pos, with_labels=False)

        
    def to_pyvis(self, n_nodes=1_000, notebook=False) -> Network:
        
        subG = self.subgraph(n_nodes)
    
        # net = Network('700px', '1000px', notebook=True)
        net = Network(height='500px', width='100%', notebook=notebook)
        net.from_nx(subG)
        
        net.force_atlas_2based()
        # net.barnes_hut(central_gravity=1, spring_length=100)
        net.show_buttons(filter_=['physics']) 
        # net.show(f'nx_{child}-{parent}.html')
        return net
    
    
    def __repr__(self):
        return repr(self.g)
    
    def __str__(self):
        return str(self.g)
    
    def __len__(self):
        return len(self.g)
    
    
    
    
# import graph_tool.all as gt



# pos = gt.sfdp_layout(wg.g, eweight=wg.g.ep.occ, 
#                      p=3, C=.5,
#                     )

# gt.graph_draw(wg.g, pos=pos,
#               output_size=(1_000, 1_000),
#               nodesfirst=True,
              
#               vertex_font_size=10,
#               vertex_pen_width=0,
#               vertex_fill_color='lightblue',
#               vertex_text=wg.g.vp.text,
             
#               edge_marker_size=10,
#               edge_text=wg.g.ep.occ,
#              )