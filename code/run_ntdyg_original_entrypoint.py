#!/usr/bin/env python3
"""Run the retained binary entrypoint, including all four original IF branches.

Only the hard-coded dataset loop is parameterized. Model/optimizer/loss,
graph construction, checkpoint selection, early stopping and IF settings use
snapshots of the original files. Retained memory weights are fixed inputs;
these are classifier/IF repetitions, not end-to-end memory-training repeats.
"""
from pathlib import Path
import argparse
import ast
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback


CODE = Path(__file__).resolve().parent
PROJECT = CODE.parent
SOURCE = PROJECT / "legacy" / "binary"
DATA_DIR = Path(os.environ.get("NTDYG_DATA_DIR", PROJECT / "data" / "binary"))
MEMORY_INPUT_DIR = Path(os.environ.get("NTDYG_MEMORY_INPUT_DIR", PROJECT / "data" / "legacy_memory"))
DATASETS = ("IIoTAPT", "Darknet", "CloudDDoS")
SEEDS = (95545, 95546, 95547, 95548)
BRANCHES = ("direct", "logit_if", "feature_if", "logit_gru_if", "feature_gru_if")
DEFAULT_ROOT = PROJECT / "outputs" / "retained_ntdyg"


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def now():
    return time.time()


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    snapshot = root / "source_snapshot"
    snapshot.mkdir()
    files = list(SOURCE.glob("*.py")) + list((SOURCE / "model").glob("*.py"))
    hashes = {}
    for original in files:
        relative = original.relative_to(SOURCE)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        hashes[str(relative)] = sha256(original)
        assert sha256(target) == hashes[str(relative)]
    (snapshot / "dataset").symlink_to(DATA_DIR, target_is_directory=True)
    inputs = {}
    for dataset in DATASETS:
        output = root / "shared" / dataset
        output.mkdir(parents=True)
        originals = [DATA_DIR / f"{dataset}_{suffix}.csv"
                     for suffix in ("re", "finetune_indx", "test_indx")]
        for filename in ("memory_updates_each_event.csv", "test_matrix.csv",
                         "final_memory_matrix.csv", "trained_GRU_memory_model.pt"):
            original = MEMORY_INPUT_DIR / dataset / filename
            if not original.is_file():
                raise FileNotFoundError(original)
            (output / filename).symlink_to(original)
            originals.append(original)
        inputs[dataset] = {str(p): dict(bytes=p.stat().st_size, sha256=sha256(p)) for p in originals}
    save(root / "protocol.json", dict(
        registered_at=now(), original_source=str(SOURCE), datasets=list(DATASETS), seeds=list(SEEDS),
        expected_runs=12, branches=list(BRANCHES), source_sha256=hashes, inputs=inputs,
        retained_settings="Original binary entrypoint, khop_1, 200 maximum epochs, 15-round early stopping",
        selection="Original test-AUC selection retained separately for each branch; early stopping uses direct AUC",
        isolation_forest="Original 200 trees, max_samples=1024, contamination=0.03, random_state=42",
        memory_scope="Retained fixed memory weights, recorded training memory and final state; no memory retraining",
        execution="Original model forward and per-graph padded loaders; no target-row reduction",
        entrypoint_change="Only replace hard-coded dataset-loop iterable with the requested dataset",
        measurement_observers="Record original evaluation scores/metrics and save matching IF estimators",
        historical_result_reproduction_verified=False, independent_test=False,
        automatically_replace_manuscript=False,
        unavailable_original_methods=["KAO", "IMLP", "ACODT", "CLSTM", "RGAN", "TRACER", "GSAGE",
                                      "GIN", "GCN", "G2VRF", "DAGMM", "DevNet", "DIF"],
        runner_sha256=sha256(__file__)))
    (root / "runtime_tmp").mkdir()
    (root / "logs").mkdir()
    print(json.dumps(dict(prepared=str(root), expected_runs=12), ensure_ascii=False), flush=True)


def original_tree(path):
    """Parameterize only the known top-level dataset iterable, with an audit."""
    tree = ast.parse(path.read_text())
    matches = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.For) and isinstance(node.target, ast.Attribute)
                and isinstance(node.target.value, ast.Name)
                and node.target.attr == "data_set"
                and node.target.value.id in ("args", "config")
                and isinstance(node.iter, ast.List)
                and all(isinstance(x, ast.Constant) and x.value in DATASETS for x in node.iter.elts)):
            matches.append(node)
    if len(matches) != 1:
        raise ValueError(f"Expected one known dataset loop in {path}, found {len(matches)}")
    loop = matches[0]
    original = ast.unparse(loop.iter)
    loop.iter = ast.List(elts=[ast.Attribute(value=ast.Name(id=loop.target.value.id, ctx=ast.Load()),
                                          attr="data_set", ctx=ast.Load())], ctx=ast.Load())
    ast.fix_missing_locations(tree)
    return tree, dict(file=path.name, line=loop.lineno, original_iterable=original,
                      new_iterable=ast.unparse(loop.iter))


def initialize(root, dataset, output, seed):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    snapshot = root / "source_snapshot"
    protocol = json.loads((root / "protocol.json").read_text())
    for name, digest in protocol["source_sha256"].items():
        assert sha256(snapshot / name) == digest, name
    assert sha256(__file__) == protocol["runner_sha256"], "Runner changed after registration"
    os.chdir(snapshot)
    sys.path.insert(0, str(snapshot))
    sys.argv = [str(snapshot / "NTDyG_V1.py"), "--data_set", dataset,
                "--seed", str(seed), "--dir_data", str(snapshot / "dataset"),
                "--output_dir", str(output)]
    return snapshot


def preprocess(root, dataset):
    output = root / "shared"
    status = output / dataset / "preprocessing_status.json"
    save(status, dict(status="running", dataset=dataset, pid=os.getpid(), started_at=now()))
    snapshot = initialize(root, dataset, output, 42)
    path = snapshot / "rawdatasets_pre.py"
    tree, change = original_tree(path)
    save(output / dataset / "entrypoint_adjustment.json", change)
    exec(compile(tree, str(path), "exec"), dict(__name__="__main__", __file__=str(path)))
    exports = {}
    for part in ("train", "test"):
        p = output / dataset / f"{dataset}_{part}_khop_1.pkl"
        assert p.is_file(), p
        exports[part] = dict(path=str(p), bytes=p.stat().st_size, sha256=sha256(p))
    save(status, dict(status="complete", dataset=dataset, finished_at=now(), exports=exports))


def train(root, dataset, seed):
    import joblib
    import numpy as np
    from sklearn.metrics import average_precision_score
    from sklearn.ensemble import IsolationForest
    run = root / "jobs" / dataset / f"seed_{seed}"
    run.mkdir(parents=True, exist_ok=False)
    output = run / "outputs"
    target = output / dataset
    target.mkdir(parents=True)
    for part in ("train", "test"):
        name = f"{dataset}_{part}_khop_1.pkl"
        (target / name).symlink_to(root / "shared" / dataset / name)
    snapshot = initialize(root, dataset, output, seed)
    import torch
    import utils
    path = snapshot / "NTDyG_V1.py"
    tree, change = original_tree(path)
    save(run / "run_metadata.json", dict(status="running", dataset=dataset, seed=seed,
         pid=os.getpid(), started_at=now(), entrypoint_adjustment=change,
         cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
         torch_version=torch.__version__, gpu=torch.cuda.get_device_name(0),
         memory_scope="fixed retained memory inputs", maximum_epochs=200,
         graph_sha256={p: sha256(root / "shared" / dataset / f"{dataset}_{p}_khop_1.pkl")
                       for p in ("train", "test")}))
    captured, forests, history, best = [], [], [], {}
    original_auc = utils.roc_auc_score
    original_evaluate = utils.eval_epoch_finetune_one

    def observe_auc(y, scores, *args, **kwargs):
        result = original_auc(y, scores, *args, **kwargs)
        captured.append((np.asarray(y).copy(), np.asarray(scores).copy(), float(result)))
        return result

    def make_forest(*args, **kwargs):
        model = IsolationForest(*args, **kwargs)
        forests.append(model)
        return model

    def observe_evaluate(*args, **kwargs):
        captured.clear()
        result = original_evaluate(*args, **kwargs)
        assert len(captured) == 5 and len(result) == 25
        epoch = len(history)
        record = dict(epoch=epoch, evaluated_at=now(), branches={})
        for index, branch in enumerate(BRANCHES):
            labels, scores, auc = captured[index]
            assert np.isfinite(scores).all(), branch
            ap = float(average_precision_score(labels, scores))
            original = list(map(float, result[5*index:5*(index+1)]))
            assert abs(auc-original[0]) < 1e-12
            assert abs(ap-original[3]) < 1e-12
            metrics = dict(roc_auc=auc, average_precision=ap, accuracy=original[1],
                           macro_precision=original[2], macro_f1=original[4])
            record["branches"][branch] = metrics
            if branch not in best or auc > best[branch]["roc_auc"]:
                directory = run / "selected" / branch
                directory.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(directory / "scores.npz", labels=labels, scores=scores)
                if index:
                    joblib.dump(forests[index-1], directory / "isolation_forest.joblib")
                best[branch] = dict(selected_epoch=epoch, **metrics)
                save(directory / "metrics.json", best[branch])
        history.append(record)
        save(run / "epoch_metrics.json", history)
        save(run / "progress.json", dict(status="running", dataset=dataset, seed=seed,
             epochs_completed=len(history), updated_at=now(), best=best))
        print(json.dumps(dict(event="recorded_original_epoch", dataset=dataset, seed=seed,
              epoch=epoch, direct_auc=best["direct"]["roc_auc"],
              feature_gru_if_auc=best["feature_gru_if"]["roc_auc"])), flush=True)
        return result

    utils.roc_auc_score = observe_auc
    utils.eval_epoch_finetune_one = observe_evaluate
    namespace = dict(__name__="original_ntdyg_observed", __file__=str(path))
    exec(compile(tree, str(path), "exec"), namespace)
    # This records original estimators; their class/settings and fit/inference remain unchanged.
    namespace["IsolationForest"] = make_forest
    namespace["main"]()
    save(run / "results.json", dict(status="complete", dataset=dataset, seed=seed,
         epochs_completed=len(history), best=best, finished_at=now(),
         original_rec_log_field_is="average precision, not recall",
         independent_test=False, historical_result_reproduction_verified=False,
         memory_scope="fixed retained memory inputs"))
    save(run / "progress.json", dict(status="complete", dataset=dataset, seed=seed,
         epochs_completed=len(history), updated_at=now(), best=best))


def child(root, action, dataset, gpu, log, seed=None):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), TMPDIR=str(root / "runtime_tmp"),
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               LOKY_MAX_CPU_COUNT="8", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    command = [sys.executable, "-u", str(Path(__file__).resolve()), action,
               "--root", str(root), "--dataset", dataset]
    if seed is not None:
        command += ["--seed", str(seed)]
    with Path(log).open("w") as stream:
        process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
    return process, command


def pipeline(root, dataset, gpu):
    state = root / "pipeline_status" / f"{dataset}.json"
    prep = root / "shared" / dataset / "preprocessing_status.json"
    if not prep.exists() or json.loads(prep.read_text())["status"] != "complete":
        log = root / "logs" / f"{dataset}_preprocess.log"
        process, command = child(root, "preprocess", dataset, gpu, log)
        save(state, dict(status="preprocessing", dataset=dataset, pid=process.pid,
                         gpu=gpu, log=str(log), command=command, updated_at=now()))
        if process.wait():
            raise RuntimeError(f"Preprocessing failed; see {log}")
    for seed in SEEDS:
        run = root / "jobs" / dataset / f"seed_{seed}"
        if (run / "results.json").is_file():
            continue
        if run.exists():
            raise FileExistsError(f"Incomplete run retained for inspection: {run}")
        log = root / "logs" / f"{dataset}_seed_{seed}.log"
        process, command = child(root, "train", dataset, gpu, log, seed)
        save(state, dict(status="training", dataset=dataset, seed=seed, pid=process.pid,
                         gpu=gpu, log=str(log), command=command, updated_at=now()))
        if process.wait():
            raise RuntimeError(f"Original training failed; see {log}")
    save(state, dict(status="complete", dataset=dataset, completed_runs=4, updated_at=now()))


def summarize(root):
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score
    records = []
    for path in sorted((root / "jobs").glob("*/seed_*/results.json")):
        result = json.loads(path.read_text())
        for branch, metrics in result["best"].items():
            with np.load(path.parent / "selected" / branch / "scores.npz") as stored:
                auc = float(roc_auc_score(stored["labels"], stored["scores"]))
                ap = float(average_precision_score(stored["labels"], stored["scores"]))
            assert abs(auc-metrics["roc_auc"]) < 1e-12
            assert abs(ap-metrics["average_precision"]) < 1e-12
            records.append(dict(dataset=result["dataset"], seed=result["seed"], branch=branch,
                                roc_auc=100*auc, average_precision=100*ap))
    groups = []
    for dataset in DATASETS:
        for branch in BRANCHES:
            rows = [r for r in records if r["dataset"] == dataset and r["branch"] == branch]
            if len(rows) != 4:
                continue
            groups.append(dict(dataset=dataset, branch=branch, seeds=len(rows),
                **{metric+suffix:float(np.mean([r[metric] for r in rows]) if suffix=="_mean"
                         else np.std([r[metric] for r in rows], ddof=1))
                   for metric in ("roc_auc", "average_precision") for suffix in ("_mean", "_sample_sd")}))
    if groups:
        with (root / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(groups[0]))
            writer.writeheader(); writer.writerows(groups)
    save(root / "score_audit.json", dict(verified_metric_rows=len(records), complete_groups=len(groups),
                                        expected_groups=15, results=records))
    return len(records)//5, groups


def queue(root, gpus):
    if len(gpus) != 3:
        raise ValueError("Specify three unoccupied GPUs for the three dataset pipelines")
    processes = []
    for dataset, gpu in zip(DATASETS, gpus):
        log = root / "logs" / f"{dataset}_pipeline.log"
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "pipeline", "--root", str(root),
                   "--dataset", dataset, "--gpu", str(gpu)]
        with log.open("w") as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        processes.append((dataset, process))
    while True:
        states = []
        for dataset, process in processes:
            path = root / "pipeline_status" / f"{dataset}.json"
            record = json.loads(path.read_text()) if path.is_file() else dict(dataset=dataset, status="starting")
            record.update(pipeline_pid=process.pid, return_code=process.poll())
            states.append(record)
        completed = len(list((root / "jobs").glob("*/seed_*/results.json")))
        done = all(process.poll() is not None for _, process in processes)
        status = ("complete" if completed == 12 else "failed") if done else "running"
        save(root / "queue_status.json", dict(status=status, manager_pid=os.getpid(), updated_at=now(),
             expected_runs=12, completed_runs=completed, pipelines=states))
        if done:
            summarize(root)
            if status != "complete":
                raise RuntimeError("One or more original-code pipelines failed; see preserved logs")
            break
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "preprocess", "train", "pipeline", "queue", "summarize"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpus", nargs=3, type=int, default=[0, 1, 2])
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        if args.action == "prepare": prepare(root)
        elif args.action == "preprocess": preprocess(root, args.dataset)
        elif args.action == "train": train(root, args.dataset, args.seed)
        elif args.action == "pipeline": pipeline(root, args.dataset, args.gpu)
        elif args.action == "queue": queue(root, args.gpus)
        elif args.action == "summarize": summarize(root)
    except Exception as error:
        save(root / "errors" / f"{args.action}_{args.dataset}_{args.seed}_{os.getpid()}.json",
             dict(action=args.action, dataset=args.dataset, seed=args.seed, error=str(error),
                  traceback=traceback.format_exc(), failed_at=now()))
        raise


if __name__ == "__main__":
    main()
