"""Measure actual event-to-score execution of the registered direct-score pipeline."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import gc
import json
import os
import resource
import sys
import threading
import time

import numpy as np
import pandas as pd
import psutil
try:
    import pynvml
except ImportError:
    pynvml = None
import torch

ROOT=Path(__file__).resolve().parent
CLI_ARGS=sys.argv[1:]
from stream_graph import c, Event, IncrementalGraph, batch_from_graph, events_from_frame


def restore_state(run,valid):
    state=torch.load(run/'train_terminal_memory.pt',map_location='cpu',weights_only=True)
    memory,last=state['memory'].clone(),state['last_update'].clone()
    src=valid[[f'src_mem_{j}' for j in range(16)]].to_numpy(dtype=np.float32,copy=True)
    dst=valid[[f'dst_mem_{j}' for j in range(16)]].to_numpy(dtype=np.float32,copy=True)
    for i,row in enumerate(valid[['src_id','dst_id','time']].itertuples(index=False,name=None)):
        s,d,t=int(row[0]),int(row[1]),float(row[2])
        memory,last=c.extend_observed_state(memory,last,s,d)
        memory[s]=torch.from_numpy(src[i]);memory[d]=torch.from_numpy(dst[i])
        last[s]=t;last[d]=t
    return memory,last


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['CloudDDoS','IIoTAPT','Darknet'], default='CloudDDoS')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--repeat',type=int,default=1)
    parser.add_argument('--verify-events',type=int,default=0,
                        help='Correctness preflight only; save no performance measurements.')
    args=parser.parse_args(CLI_ARGS)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    run=args.run_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result=json.loads((run/'results.json').read_text())
    meta=json.loads((run/'run_metadata.json').read_text())
    device=torch.device(args.device);config=c.DATASET_CONFIG[args.dataset];cols=meta['feature_columns']
    if device.type=='cuda':torch.cuda.set_device(device)
    raw=pd.read_csv(meta['source_csv'])
    indices={name:c.read_indices(Path(meta['split_dir'])/f'{name}_indices.csv') for name in ['train','validation','test']}
    trainraw=raw.iloc[indices['train']]
    means,stds=c.fit_standardizer(trainraw,cols)
    means=means.to_numpy(dtype=np.float64);stds=stds.replace(0,1.).to_numpy(dtype=np.float64)
    registry={}
    for name in ['train','validation']:
        for s,d in raw.iloc[indices[name]][['Src IP Label','Dst IP Label']].itertuples(index=False,name=None):
            for x in [int(s),int(d)]:
                if x not in registry:registry[x]=len(registry)
    train=pd.read_pickle(run/'train_event_features.pkl');valid=pd.read_pickle(run/'validation_event_features.pkl')
    initial_memory,initial_last=restore_state(run,valid)
    initial_graph=IncrementalGraph(len(cols),meta['training_entity_count']-1)
    for frame in [train,valid]:
        for event in events_from_frame(frame,cols):initial_graph.commit(event)
    test=raw.iloc[indices['test']]
    rawfeatures=test[cols].to_numpy(dtype=np.float64)
    endpoints=test[['Src IP Label','Dst IP Label']].to_numpy(dtype=np.int64)
    timestamps=test.Timestamp_Label.to_numpy(dtype=np.float64)
    labels=test.Label_encoded.to_numpy(dtype=np.int64)
    del train,valid,raw,test,trainraw;gc.collect()
    memory_model=c.NodeMemoryModel(num_nodes=meta['training_entity_count'],feature_dim=len(cols),memory_dim=16,
         message_dim=len(cols),hidden_dim=32,classifier_hidden_dim=32,device='cpu')
    memory_model.load_state_dict(torch.load(run/'memory_model.pt',map_location='cpu',weights_only=True));memory_model.eval()
    cfg=SimpleNamespace(num_classes=2,input_dim=len(cols),n_heads=config['heads'],drop_out=.2,n_layer=config['layers'])
    classifier=c.CombinedModel(cfg,c.CensNet(17,len(cols),.2),
        c.TransformerTargetEdgeClassifier(cfg,device,hidden_size=config['hidden']),node_memory_dim=16).to(device)
    classifier.load_state_dict(torch.load(run/'best_validation_checkpoint.pt',map_location=device,weights_only=True));classifier.eval()
    threshold=result['validation_threshold_selected_by_f1']
    process=psutil.Process()

    @torch.inference_mode()
    def execute(limit,measure):
        memory,last=initial_memory.clone(),initial_last.clone()
        identities=registry.copy()
        graph=IncrementalGraph(len(cols),meta['training_entity_count']-1)
        for key,records in initial_graph.incoming.items():graph.incoming[key].extend(records)
        scores=[];latencies=[];resources=[]
        start_all=time.perf_counter()
        for i in range(limit):
            started=time.perf_counter()
            feature=((rawfeatures[i]-means)/stds).astype(np.float32)
            for entity in endpoints[i]:
                if int(entity) not in identities:identities[int(entity)]=len(identities)
            s,d=(identities[int(x)] for x in endpoints[i])
            memory,last=c.extend_observed_state(memory,last,s,d)
            src_new,dst_new,_,_,_=memory_model.compute_updated_states_and_score(
                s,d,float(timestamps[i]),torch.from_numpy(feature),memory,last)
            event=Event(int(indices['test'][i]),s,d,feature,src_new.numpy().copy(),dst_new.numpy().copy())
            batch=batch_from_graph(graph.build(event))
            score=float(c.run_model(classifier,batch,device)[0,1].item())
            alert=score>=threshold
            memory[s]=src_new;memory[d]=dst_new;last[s]=float(timestamps[i]);last[d]=float(timestamps[i])
            graph.commit(event)
            elapsed=time.perf_counter()-started
            if measure:
                scores.append(score);latencies.append(elapsed*1000)
                if i%1024==0 or i==limit-1:
                    resources.append(dict(event=i+1,observed_entities=len(identities),
                         history_records=sum(len(x) for x in graph.incoming.values()),
                         rss_mib=process.memory_info().rss/2**20,
                         state_bytes=memory.numel()*memory.element_size()+last.numel()*last.element_size()))
                if i%4096==0:
                    print(json.dumps(dict(device=str(device),repeat=args.repeat,events_completed=i+1)),flush=True)
        return np.asarray(scores),np.asarray(latencies),resources,time.perf_counter()-start_all

    execute(min(32, len(labels)),False)
    if device.type=='cuda':torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
    handle=None;other_processes_start=[];other_processes_end=[]
    if device.type=='cuda':
        if pynvml is None:
            raise RuntimeError('Install nvidia-ml-py for CUDA memory profiling')
        pynvml.nvmlInit();handle=pynvml.nvmlDeviceGetHandleByIndex(device.index)
        owned_devices=[]
        for index in range(pynvml.nvmlDeviceGetCount()):
            current=pynvml.nvmlDeviceGetHandleByIndex(index)
            if any(p.pid==os.getpid() for p in pynvml.nvmlDeviceGetComputeRunningProcesses(current)):
                owned_devices.append(index)
        assert owned_devices==[device.index],f'Unexpected additional GPU contexts: {owned_devices}'
        other_processes_start=[p.pid for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle) if p.pid!=os.getpid()]
    stop=threading.Event();peaks={'rss':process.memory_info().rss,'gpu':0,'samples':0,'other_gpu_processes':0};monitor_errors=[]
    def monitor():
        try:
            while not stop.is_set():
                peaks['rss']=max(peaks['rss'],process.memory_info().rss);peaks['samples']+=1
                if handle is not None:
                    infos=pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    peaks['other_gpu_processes']=max(peaks['other_gpu_processes'],sum(info.pid!=os.getpid() for info in infos))
                    for info in infos:
                        if info.pid==os.getpid():peaks['gpu']=max(peaks['gpu'],info.usedGpuMemory)
                stop.wait(.01)
        except Exception as error:
            monitor_errors.append(repr(error))
    worker=threading.Thread(target=monitor,daemon=True);worker.start()
    try:
        limit=min(args.verify_events,len(labels)) if args.verify_events>0 else len(labels)
        scores,latencies,resources,wall=execute(limit,True)
    finally:
        stop.set();worker.join()
    assert not monitor_errors,monitor_errors
    assert peaks['samples']>0
    if handle is not None:
        assert 0<peaks['gpu']<=pynvml.nvmlDeviceGetMemoryInfo(handle).total
    if handle is not None:
        other_processes_end=[p.pid for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle) if p.pid!=os.getpid()]
    expected=pd.read_csv(run/'test_scores.csv').anomaly_score_class1_logit.to_numpy()
    if args.verify_events>0:
        expected=expected[:limit]
        np.testing.assert_allclose(scores,expected,rtol=1e-4,atol=1e-4)
        preflight=dict(passed=True,events=limit,device=str(device),seed=int(meta["seed"]),
            maximum_absolute_score_difference=float(np.max(np.abs(scores-expected))),
            threshold_decision_differences=int(np.sum((scores>=threshold)!=(expected>=threshold))),
            scope='Correctness preflight only; excluded from all latency statistics')
        (args.output_dir/f"stream_preflight_{args.device.replace(':','_')}.json").write_text(json.dumps(preflight,indent=2))
        print(json.dumps(preflight),flush=True)
        return
    np.testing.assert_allclose(scores,expected,rtol=1e-4,atol=1e-4)
    measured=c.metric_record(labels,scores,threshold)
    for key in ['roc_auc','average_precision']:
        assert abs(measured[key]-result['test_metrics'][key])<1e-4,(key,measured[key])
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    name=f"{args.device.replace(':','_')}_repeat_{args.repeat}"
    if (out/(name+'.npz')).exists() or (out/(name+'.json')).exists():
        raise FileExistsError('Use a new repeat number or output directory')
    np.savez_compressed(out/(name+'.npz'),scores=scores,labels=labels,latency_ms=latencies)
    report=dict(device=str(device),repeat=args.repeat,seed=int(meta["seed"]),events=len(labels),
        latency_mean_ms=float(latencies.mean()),latency_median_ms=float(np.median(latencies)),
        latency_p95_ms=float(np.percentile(latencies,95)),latency_p99_ms=float(np.percentile(latencies,99)),
        complete_loop_seconds=wall,throughput_events_per_second=len(labels)/wall,
        process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        streaming_peak_rss_mib=peaks['rss']/2**20,monitor_samples=peaks['samples'],monitor_interval_ms=10,
        nvml_process_peak_gpu_mib=peaks['gpu']/2**20 if handle is not None else None,
        cuda_peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None,
        cuda_peak_reserved_mib=torch.cuda.max_memory_reserved(device)/2**20 if device.type=='cuda' else None,
        maximum_absolute_score_difference=float(np.max(np.abs(scores-expected))),
        threshold_decisions_differing_from_batch=int(np.sum((scores>=threshold)!=(expected>=threshold))),
        metrics=measured,resources=resources,
        scope='Raw flow-vector standardization, observed-ID registration, memory computation, causal graph construction, batch-one classifier score, threshold decision, memory/history commit',
        excluded='Packet capture, flow-feature extraction, CSV I/O, model loading, and pre-test state restoration',
        scoring_branch='Direct classifier; this branch contains no Isolation Forest',
        hardware_scope='Shared workstation; not edge-device or attack-onset detection delay',
        other_gpu_processes_at_start=other_processes_start,other_gpu_processes_at_end=other_processes_end,
        maximum_other_gpu_processes_observed=peaks['other_gpu_processes'] if handle is not None else None,
        gpu_contention_scope='Concurrent workloads are retained and recorded; no exclusive-device speed claim',
        memory_device='CPU for both profiles; classifier device is the indicated device',
        parameter_updates=False,threshold_selection='frozen clean validation threshold',
        peak_rss_scope='Whole-process high-water mark, including setup; per-event RSS samples also retained')
    report['hardware']=dict(cpu_model=next(line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),
        system_ram_gib=psutil.virtual_memory().total/2**30,torch_threads=torch.get_num_threads(),
        gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        torch_version=torch.__version__)
    report['source_sha256']={name:c.sha256(ROOT/name) for name in ['benchmark_stream.py','stream_graph.py']}
    report['checkpoint_sha256']={name:c.sha256(run/name) for name in ['memory_model.pt','best_validation_checkpoint.pt']}
    (out/(name+'.json')).write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='resources'}),flush=True)


if __name__=='__main__':main()
