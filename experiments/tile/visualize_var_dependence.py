# %%
from typing import Dict, List
from hidet.ir import IRModule, Var
from hidet.transforms.tile.analyzers import DependencyAnalyzer
import networkx as nx
from pyvis.network import Network

def ir_to_graph(ir: IRModule) -> nx.Graph:
    analyzer = DependencyAnalyzer()
    analyzer.visit(ir)
    depends: Dict[Var, List[Var]] = analyzer.depends
    G = nx.DiGraph()
    for k, v in depends.items():
        G.add_node(str(k))
        for i in v:
            G.add_edge(str(k), str(i))
    return G

def ir_to_graph_html(ir: IRModule, path: str, notebook=False):
    g = ir_to_graph(ir)
    net = Network(notebook=notebook, directed=True)
    net.from_nx(g)
    net.show(path)


def dependence_graph_to_graph_html(deps: Dict[Var, List[Var]], path: str, notebook=False):
    G = nx.DiGraph()
    for k, v in deps.items():
        G.add_node(str(k))
        for i in v:
            G.add_edge(str(k), str(i))
    
    net = Network(notebook=notebook, directed=True)
    net.from_nx(G)
    net.show(path)

