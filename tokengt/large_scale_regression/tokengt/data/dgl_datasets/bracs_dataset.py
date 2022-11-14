import torch
import os
import dgl
import traceback
from torch_geometric.utils import from_networkx as pyg_from_networkx
from dgl.data import DGLDataset
from dgl.data.utils import load_graphs
import gc


class TokenGTDGLLocalDataset(DGLDataset):
    def __init__(self, abs_path):
        self.root_path = abs_path
        super().__init__(name='synthetic')

    def process(self):
        self.graphs = []
        self.labels = []
        dirs = ["train", "test", "val"]
        sets =  []
        idx = 0
        print("Loading dataset ...")
        for i, dir in enumerate(dirs):
            sets.append([])
            path = os.path.join(self.root_path, dir)
            for file in os.listdir(path):
                if file.endswith(".bin"):
                    file_path = os.path.join(path, file)
                    try:
                        g = load_graphs(file_path)
                        # g_pyg = pyg_from_networkx(g[0][0].to_networkx(node_attrs=["centroid", "feat"]))
                        # g_pyg.y = torch.tensor([g[1]["label"].item()])
                        # g_pyg.x = g_pyg.feat
                        # g_pyg.edge_attr = torch.stack([g_pyg.x.index_select(0, indices).mean(dim=0) for indices in g_pyg.edge_index.T])
                        # self.graphs.append(g_pyg)
                        # sets[i].append(g_pyg)

                        graph = g[0][0]
                        graph.y = g[1]["label"]
                        self.graphs.append(graph)
                        sets[i].append(graph)
                        
                        idx += 1
                        del g, graph
                        gc.collect()
                    except Exception as e:
                        pass
                        # traceback.print_exc()

        self.train_set, self.valid_set, self.test_set = sets
        print("Done Loading dataset.")

    def __getitem__(self, i):
        return self.graphs[i]

    def __len__(self):
        return len(self.graphs)