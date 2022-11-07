from math import cos, sin, pi
from networkx import Graph, DiGraph, layout, draw, draw_networkx_labels as draw_labels,  draw_networkx_edge_labels as draw_edge_labels
from IPython.display import HTML

def draw_graph(graph, ax=None, node_labels=None):
    n = len(graph["nodes"])
    m = n // 5
    pos = {i: (cos(-(i*2*pi*m) / n + 0.5*pi) / (i * m//n + 1), 
               sin(-(i*2*pi*m) / n + 0.5*pi) / (i * m//n + 1)) 
           for i in graph["nodes"]}
    colors = [ord(node_labels[i].upper()) - 65 for i in graph["nodes"]] if node_labels is not None else None
    draw(Graph(graph["edges"]), pos=pos, ax=ax, with_labels=True, 
         font_color="white", vmin=0, vmax=10, cmap="tab10", node_color=colors)


def draw_network(network, ax=None, edge_flows=None):
    g = DiGraph(network["edges"].keys())
    pos = layout.kamada_kawai_layout(g, weight=None)
    draw(g, pos=pos, ax=ax, with_labels=True, font_color="white")
    if edge_flows is None:
        draw_labels(g, ax=ax, font_color="red", font_weight="bold",
                    pos={k: v - (0,0.08) for k, v in pos.items()}, 
                    labels={i: ",".join(f"{k}={v}" for k, v in data.items()) 
                            for i, data in network["nodes"].items()})
        draw_edge_labels(g, pos=pos, ax=ax, font_size=9,
                         edge_labels={i: ",".join(f"{k}={v}" for k, v in data.items()) 
                                      for i, data in network["edges"].items()})
    else:
        draw_edge_labels(g, pos=pos, ax=ax, font_size=11, font_weight="bold", edge_labels=edge_flows)


def display_side_by_side(dfs:list, captions:list):
    output = ""
    combined = dict(zip(captions, dfs))
    for caption, df in combined.items():
        output += df.style.set_table_attributes("style='display:inline'").set_caption(caption)._repr_html_()
        output += "\xa0\xa0\xa0"
    display(HTML(output))
