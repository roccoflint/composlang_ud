
from collections import defaultdict

import graph_tool.all as gt
import networkx as nx

class WordGraph():
        
    def __init__(self, wordpairs, backend='nx'):
        
        if backend == 'gt':
            g = gt.Graph()

            g.ep.occ = g.new_ep("int")
            g.vp.upos = g.new_vp("string")
            g.vp.text = g.new_vp("string")

            self.vertices = defaultdict(lambda: g.add_vertex()) # word, upos -> vertex
            self.edges = dict() # word, upos -> vertex
            for w, p in wordpairs:

                w_ = self.vertices[w.text, w.upos]
                p_ = self.vertices[p.text, p.upos]

                edge_name = w.text, w.upos, p.text, p.upos
                if edge_name not in self.edges:
                    self.edges[edge_name] = g.add_edge(w_, p_)
                e = self.edges[edge_name]

                g.vp.text[w_], g.vp.upos[w_] = w.text, w.upos
                g.vp.text[p_], g.vp.upos[p_] = p.text, p.upos

                g.ep.occ[e] += 1 
                
        elif backend == 'nx':
            
            g = nx.DiGraph()
            nodes = {} # tuple -> int
            edges = defaultdict(int)
            
            for w, p in wordpairs:
                w_ = (w.text,w.upos)
                p_ = (p.text,p.upos)
                
                nodes[w_] = nodes.get(w_, len(nodes))
                nodes[p_] = nodes.get(p_, len(nodes))
                
                g.add_node(nodes[w_], label=w.text, group=w.upos)
                g.add_node(nodes[p_], label=p.text, group=p.upos)
                
                edges[nodes[w_], nodes[p_]] += 1
                
            for edge, numocc in edges.items():
                g.add_edge(*edge, label=numocc)
                
        self.g = g
        
    def __repr__(self):
        return repr(self.g)
    
    def __str__(self):
        return str(self.g)
    
    
    
    
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