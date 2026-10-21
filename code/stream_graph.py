"""Incremental k=1, degree=10 graph construction using the original graph operations."""
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
CLI_ARGS=sys.argv[1:]
sys.path.insert(0,str(ROOT))
import run_candidate_experiment as c


@dataclass
class Event:
    index: int
    src: int
    dst: int
    feature: np.ndarray
    src_memory: np.ndarray
    dst_memory: np.ndarray


class IncrementalGraph:
    def __init__(self, feature_count, max_train_id):
        self.feature_count=feature_count
        self.max_train_id=max(max_train_id,1)
        self.incoming=defaultdict(lambda:deque(maxlen=10))

    def commit(self,event):
        self.incoming[event.dst].append(event)

    def build(self,event):
        # The target participates in its own graph, including a possible self-loop.
        predecessors=list(self.incoming.get(event.src,()))
        if event.src==event.dst:
            predecessors=(predecessors+[event])[-10:]
        records=predecessors if predecessors and predecessors[-1] is event else predecessors+[event]
        sampler=c.TemporalBatchGraphSample.__new__(c.TemporalBatchGraphSample)
        sampler.k_hop=1;sampler.n_degree=10;sampler.max_node_id=self.max_train_id
        sampler.src_node_feature_cols=list(range(16));sampler.edge_feature_cols=list(range(self.feature_count))
        sampler.full_src=np.asarray([r.src for r in records],dtype=np.int64)
        sampler.full_dst=np.asarray([r.dst for r in records],dtype=np.int64)
        sampler.full_src_node_feat=np.asarray([r.src_memory for r in records],dtype=np.float32)
        sampler.full_dst_node_feat=np.asarray([r.dst_memory for r in records],dtype=np.float32)
        sampler.full_edge_feat=np.asarray([r.feature for r in records],dtype=np.float32)
        sampler.reverse_edge_rows={};sampler.reverse_edge_srcs={}
        for i,r in enumerate(records):
            sampler.reverse_edge_rows.setdefault(r.dst,[]).append(i)
            sampler.reverse_edge_srcs.setdefault(r.dst,[]).append(r.src)
        return sampler._process_event_by_global_id(len(records)-1)


def batch_from_graph(graph):
    nodes,edges,tmat,adj,eadj,_,target=graph
    n,e=len(nodes),len(edges)
    return dict(input_nodes_feature=torch.as_tensor(nodes).unsqueeze(0),
        input_edges_feature=torch.as_tensor(edges).unsqueeze(0),
        Tmats=tmat.to_dense().unsqueeze(0),adjs=adj.to_dense().unsqueeze(0),
        eadjs=eadj.to_dense().unsqueeze(0),edge_mask=torch.zeros((1,e),dtype=torch.long),
        labels=torch.zeros(1),num_nodes=torch.tensor([n]),num_edges=torch.tensor([e]),
        target_edge_index=torch.tensor([target]))


def events_from_frame(frame,columns):
    features=frame[columns].to_numpy(dtype=np.float32)
    srcmem=frame[[f'src_mem_{j}' for j in range(16)]].to_numpy(dtype=np.float32)
    dstmem=frame[[f'dst_mem_{j}' for j in range(16)]].to_numpy(dtype=np.float32)
    for i,row in enumerate(frame[['event_index','src_id','dst_id']].itertuples(index=False,name=None)):
        yield Event(int(row[0]),int(row[1]),int(row[2]),features[i].copy(),srcmem[i].copy(),dstmem[i].copy())


def verify_real_graphs(parent, output):
    import json,pickle
    import pandas as pd
    if output.exists():
        raise FileExistsError('Select a new audit output filename')
    meta=json.loads((parent/'run_metadata.json').read_text());cols=meta['feature_columns']
    train=pd.read_pickle(parent/'train_event_features.pkl')
    valid=pd.read_pickle(parent/'validation_event_features.pkl').iloc[:256]
    builder=IncrementalGraph(len(cols),int(max(train.src_id.max(),train.dst_id.max())))
    for event in events_from_frame(train,cols):builder.commit(event)
    with (parent/'graphs/validation.pkl').open('rb') as stream: reference=pickle.load(stream)
    for i,event in enumerate(events_from_frame(valid,cols)):
        graph=builder.build(event)
        for j,key in enumerate(['nodefeatures','edgefeatures','Tmats','adjs','eadjs']):
            a,b=graph[j],reference[key][i]
            if isinstance(a,torch.Tensor):
                torch.testing.assert_close(a.to_dense(),b.to_dense(),rtol=0,atol=0)
            else:np.testing.assert_array_equal(a,b)
        assert graph[-1]==reference['target_edge_indices'][i]
        builder.commit(event)
    # Explicit self-loop and repeated-edge fixtures exercise truncation and deduplication.
    rows=[]
    for i in range(25):
        row=dict(event_index=i,label=i%2,time=float(i),src_id=0 if i%3 else 1,dst_id=0,
                 f0=float(i),gru_anomaly_score=.5)
        row.update({f'{side}_mem_{j}':float(i+j) for side in ['src','dst'] for j in range(16)})
        rows.append(row)
    fixture=pd.DataFrame(rows)
    expected=c.make_graph_data(type('Config',(),dict(k_hop=1,n_degree=10,max_node_id=1))(),fixture,None,['f0'])
    builder=IncrementalGraph(1,1)
    for i,event in enumerate(events_from_frame(fixture,['f0'])):
        graph=builder.build(event)
        for j,key in enumerate(['nodefeatures','edgefeatures','Tmats','adjs','eadjs']):
            a,b=graph[j],expected[key][i]
            if isinstance(a,torch.Tensor):torch.testing.assert_close(a.to_dense(),b.to_dense(),rtol=0,atol=0)
            else:np.testing.assert_array_equal(a,b)
        builder.commit(event)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(passed=True,real_graphs=len(valid),
        self_loop_and_repeat_fixture_events=25,all_features_and_adjacencies_exactly_equal=True),indent=2))
    print(f'Verified {len(valid)} cached graphs and 25 self-loop/repeat fixtures exactly.')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Compare incremental graphs with cached validation graphs')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments=parser.parse_args(CLI_ARGS)
    torch.set_num_threads(1)
    verify_real_graphs(arguments.run_dir, arguments.output)
