"""CPU checkpoint regression tests using real Torch serialization and Adam."""
import importlib.util
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "codes" / "run.py"
METADATA = {"step": 23, "current_learning_rate": 0.0125, "warm_up_steps": 90}


class TinyModel(torch.nn.Module):
    evaluations = []

    def __init__(self, **kwargs):
        super().__init__()
        self.entity_embedding = torch.nn.Parameter(torch.tensor([[1., 2.]]))
        self.relation_embedding = torch.nn.Parameter(torch.tensor([[3., 4.]]))

    @staticmethod
    def test_step(model, triples, all_triples, args):
        TinyModel.evaluations.append((model.state_dict(), triples, args))
        return {"MRR": 0.5}


@pytest.fixture
def runner(monkeypatch):
    # Only the model and training data loader are substituted. Execute the
    # actual run.py module, including parsing, restore and main.
    from types import SimpleNamespace

    monkeypatch.setitem(sys.modules, "model", SimpleNamespace(KGEModel=TinyModel))
    monkeypatch.setitem(sys.modules, "dataloader", SimpleNamespace(
        TrainDataset=None, BidirectionalOneShotIterator=None,
    ))
    spec = importlib.util.spec_from_file_location("checkpoint_runner", RUN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Avoid altering the pytest process's logging handlers.
    monkeypatch.setattr(module, "set_logger", lambda args: None)
    TinyModel.evaluations.clear()
    return module


def write_triples(data):
    data.mkdir()
    (data / "entities.dict").write_text("0\ta\n1\tb\n", encoding="utf-8")
    (data / "relations.dict").write_text("0\tr\n", encoding="utf-8")
    for split, triple in {"train": "a\tr\ta", "valid": "a\tr\tb", "test": "b\tr\ta"}.items():
        (data / (split + ".txt")).write_text(triple + "\n", encoding="utf-8")


def save_checkpoint(runner, root, data):
    (root / "checkpoint").mkdir(parents=True)
    args = runner.parse_args([
        "-save", str(root), "--data_path", str(data), "--hp", "11", "--rp", "3",
        "--r_weight", "0.1", "--Linear_2", "-g", "9",
    ])
    model = TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=METADATA["current_learning_rate"])
    sum(p.square().sum() for p in model.parameters()).backward()
    optimizer.step()
    runner.save_model(model, optimizer, METADATA, args)
    return model, optimizer


def test_real_torch_round_trip(runner, tmp_path):
    root = tmp_path / "run"
    original, optimizer = save_checkpoint(runner, root, tmp_path / "data")
    assert (root / "checkpoint" / "checkpoint").is_file()
    args = runner.parse_args(["-init", str(root), "--do_train"])
    checkpoint = runner.load_checkpoint(args)
    restored = TinyModel()
    restored.load_state_dict(checkpoint["model_state_dict"])
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.8)
    restored_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    for name, param in original.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], param)
    expected = optimizer.state_dict()
    actual = restored_optimizer.state_dict()
    assert actual["param_groups"] == expected["param_groups"]
    for i, state in expected["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(actual["state"][i][name], value)
    assert {key: checkpoint[key] for key in METADATA} == METADATA
    assert set(checkpoint) == {*METADATA, "model_state_dict", "optimizer_state_dict"}
    assert (root / "entity_embedding.npy").is_file()
    assert (root / "relation_embedding.npy").is_file()


@pytest.mark.parametrize("mode,triple", [("valid", (0, 0, 1)), ("test", (1, 0, 0))])
@pytest.mark.parametrize("flat", [False, True])
def test_eval_only_restores_model_config_and_step(runner, tmp_path, caplog, mode, triple, flat):
    root = tmp_path / "run"
    data = tmp_path / "data"
    write_triples(data)
    original, _ = save_checkpoint(runner, root, data)
    if flat:
        nested = root / "checkpoint" / "checkpoint"
        nested.rename(root / "flat-payload")
        nested.parent.rmdir()
        (root / "flat-payload").rename(root / "checkpoint")
        config = json.loads((root / "config.json").read_text())
        for key in ["hp", "rp", "r_weight", "Linear_2", "gamma"]:
            del config[key]
        (root / "config.json").write_text(json.dumps(config))
    args = runner.parse_args(["--do_" + mode, "-init", str(root)])
    with caplog.at_level(logging.INFO):
        runner.main(args)
    assert args.save_path is None
    [(state, triples, restored_args)] = TinyModel.evaluations
    for name, value in original.state_dict().items():
        torch.testing.assert_close(state[name], value)
    assert triples == [triple]
    assert (restored_args.hp, restored_args.rp, restored_args.r_weight, restored_args.Linear_2, restored_args.gamma) == (
        (2, 5, 1, False, 12.) if flat else (11., 3., .1, True, 9.)
    )
    assert f"{mode.title()} MRR at step 23" in caplog.text


@pytest.mark.parametrize("problem,error,message", [
    ("config", FileNotFoundError, r"config\.json"),
    ("payload", FileNotFoundError, "checkpoint"),
    ("invalid-config", ValueError, r"config\.json"),
    ("missing-step", ValueError, "step"),
    ("invalid-file", ValueError, "checkpoint"),
])
def test_invalid_checkpoints_report_path(runner, tmp_path, problem, error, message):
    root = tmp_path / "run"
    payload = root / "checkpoint" / "checkpoint"
    write_triples(tmp_path / "data")
    save_checkpoint(runner, root, tmp_path / "data")
    if problem == "config":
        (root / "config.json").unlink()
    elif problem == "payload":
        payload.unlink()
    elif problem == "invalid-config":
        (root / "config.json").write_text("{")
    elif problem == "missing-step":
        torch.save({"model_state_dict": TinyModel().state_dict()}, payload)
    else:
        payload.write_bytes(b"invalid checkpoint data")
    with pytest.raises(error, match=message) as exc:
        runner.main(runner.parse_args(["--do_test", "-init", str(root)]))
    assert str(root) in str(exc.value).replace("\\\\", "\\")


def test_training_output_prefix_and_ambiguity(runner, tmp_path, monkeypatch):
    prefix = tmp_path / "model with spaces"
    args = runner.parse_args(["--hp", "11", "--rp", "5", "--r_weight", ".1"])
    path = runner.training_output_path(str(prefix), args)
    assert path == str(prefix) + "_hp: 11.0_rp: 5.0_r_weight: 0.1"
    other = str(prefix) + "_hp: 2.0_rp: 5.0_r_weight: 0.1"
    if os.name == "nt":
        # NTFS cannot create directories containing the legacy colons.
        matches = [path]
        monkeypatch.setattr(runner.glob, "glob", lambda pattern: matches.copy())
        monkeypatch.setattr(runner.os.path, "isdir", lambda directory: directory in matches)
    else:
        Path(path).mkdir()
    assert runner.resolve_checkpoint_dir(str(prefix)) == path
    if os.name == "nt":
        matches.append(other)
    else:
        Path(other).mkdir()
    with pytest.raises(ValueError, match="Multiple checkpoint"):
        runner.resolve_checkpoint_dir(str(prefix))
    assert runner.resolve_checkpoint_dir(path) == path


@pytest.mark.parametrize("mode", ["valid", "test"])
def test_shell_restores_run(runner, tmp_path, monkeypatch, caplog, mode):
    # Capture the real script's argv, then drive the real parser and main
    # without CUDA or a knowledge-graph workload.
    prefix = tmp_path / "models" / "TransE_tiny_run one"
    shell = os.environ.get("RSCF_TEST_SHELL") or (shutil.which("sh") if os.name != "nt" else None)
    if shell is None:
        pytest.skip("Requires a POSIX shell (set RSCF_TEST_SHELL for Git Bash)")
    output = prefix if os.name == "nt" else Path(str(prefix) + "_hp: 11.0_rp: 3.0_r_weight: 0.1")
    data = tmp_path / "data"
    write_triples(data)
    save_checkpoint(runner, output, data)
    shims = tmp_path / "bin"
    shims.mkdir()
    wrapper = shims / "python"
    wrapper.write_text(
        '#!/bin/sh\nif [ "$2" != "-c" ]; then printf "%s\\n" "$@" > "$RSCF_TEST_ARGS"; fi\n',
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    captured = tmp_path / "args.txt"
    result = subprocess.run(
        [shell, (ROOT / "run.sh").as_posix(), mode, "TransE", "tiny", "0", "run one"],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
        env={**os.environ, "PATH": str(shims) + os.pathsep + os.environ["PATH"],
             "RSCF_TEST_ARGS": captured.as_posix()},
    )
    assert result.returncode == 0, result.stderr
    command = captured.read_text().splitlines()
    assert command == ["-u", "codes/run.py", "--do_" + mode, "--cuda", "-init", "models/TransE_tiny_run one"]
    monkeypatch.chdir(tmp_path)
    args = runner.parse_args([arg for arg in command[2:] if arg != "--cuda"])
    with caplog.at_level(logging.INFO):
        runner.main(args)
    assert Path(args.init_checkpoint).resolve() == output
    assert f"{mode.title()} MRR at step 23" in caplog.text
