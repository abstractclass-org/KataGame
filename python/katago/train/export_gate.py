"""KataGame fork (abscls-8mbo, abscls-ehuh): the trainer's decision to export a network, in one code path.

train.py asks for an export at the end of each export cycle (-epochs-per-export). ExportDecider then applies two
guards, in this order:

1. The pace (-export-min-minutes): at most one export each this many minutes of wall-clock time. The export of a
   cycle that ends sooner after the last export is skipped, and the trainer trains on: the next cycle's export, with
   the newest weights, goes at its time. So every export stays at the end of its cycle, and in normal running, where
   the cycles end further apart, nothing changes. A trainer that catches up on a full bucket after a stop exports at
   each cycle without it: bpj5's did every 2.6 minutes, and at each new network self-play holds two networks on the
   GPU (games/blueprints/docs/decisions/port.md, "the trainer's export gate"). A pace that held a skipped export for
   the first epoch after its time would move the exports off the cycles' ends: on bpj5's log, the next cycles then
   ended less than the pace after each held export, and every export of normal running came an epoch late.
2. The gate (-export-gate-margin): the candidate against the last exported network, on the same validation rows with
   the same symmetries, in the pass of the trainer's own validation. So a change of the data (a new setting of
   self-play) changes both networks' losses, and only a change of the network shows. The gate compares both weights
   that an export writes: the raw weights, from which the training goes on, and the SWA weights, which the export
   writes as the network (export_model_pytorch.py -use-swa). It refuses the export when the candidate's p0loss or
   p1loss is above the reference's by more than the margin, for either weights. A refusal is a log line, and the
   training goes on. After -export-gate-refusals refusals in a row, the trainer halts (ExportGateError), so that a
   person looks.

The reference is the last exported network. The trainer keeps it in its folder (REFERENCE_FILE), so that a restart
compares with the same network. Without that file, the reference is the checkpoint that the trainer starts from.
"""
import logging
import math
import os
import time

import torch

REFERENCE_FILE = "export_reference.ckpt"
LOSSES = ("p0loss", "p1loss")
# The weights that an export writes, each compared with the same weights of the reference: (candidate, reference).
PAIRS = (("raw", "raw_reference"), ("swa", "swa_reference"))


class ExportGateError(RuntimeError):
    """Raised to halt a trainer whose exports the gate refused too many times in a row."""


def chunks(batch, rows):
    """The batch's rows in parts of at most ``rows`` rows: the same dict, each tensor of the batch's rows sliced."""
    n = batch["binaryInputNCHW"].shape[0]
    for lo in range(0, n, rows):
        yield {k: (v[lo:lo + rows] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == n else v)
               for k, v in batch.items()}


class GateValidation:
    """The policy losses of the gate's networks on the same validation batches.

    ``networks`` maps a name to a network that runs here on each batch (evaluate). ``forward(network, part)`` gives the
    network's outputs on a part of a batch (the trainer's autocast and inputs), and ``metrics(network, outputs, part)``
    the metrics of the trainer's validation (Metrics.metrics_dict_batchwise), whose p0loss_sum and p1loss_sum are sums
    over the part's rows. A network runs on parts of at most ``chunk_rows`` rows, so that it holds the activations of
    that many rows at a time. The sums of the parts are the sums of the batch. The trainer's own network gives its
    metrics of the whole batch with add.
    """

    def __init__(self, networks, forward, metrics, chunk_rows):
        self.networks = networks
        self.forward = forward
        self.metrics = metrics
        self.chunk_rows = max(1, int(chunk_rows))
        self.sums = {}

    def add(self, name, metrics, rows):
        s = self.sums.setdefault(name, {"rows": 0, **{k: 0.0 for k in LOSSES}})
        for k in LOSSES:
            s[k] += float(metrics[k + "_sum"])
        s["rows"] += rows

    def evaluate(self, batch):
        with torch.no_grad():
            for name, network in self.networks.items():
                was_training = network.training
                network.eval()
                try:
                    for part in chunks(batch, self.chunk_rows):
                        outputs = self.forward(network, part)
                        self.add(name, self.metrics(network, outputs, part), part["binaryInputNCHW"].shape[0])
                        del outputs
                finally:
                    network.train(was_training)

    def losses(self):
        """{name: {"p0loss": ..., "p1loss": ..., "rows": N}}, each loss per row as the trainer logs it."""
        return {name: {**{k: s[k] / s["rows"] for k in LOSSES}, "rows": s["rows"]}
                for name, s in self.sums.items() if s["rows"] > 0}


def compare(losses, margin):
    """(refused, lines): each pair of PAIRS that ``losses`` holds, the candidate's losses against the reference's.
    Refused when a candidate's p0loss or p1loss is above the reference's by more than ``margin``, or is not finite."""
    refused, lines = False, []
    for candidate, reference in PAIRS:
        if candidate not in losses or reference not in losses:
            continue
        c, r = losses[candidate], losses[reference]
        parts = []
        for k in LOSSES:
            rise = c[k] - r[k]
            bad = not math.isfinite(c[k]) or rise > margin
            refused = refused or bad
            parts.append(f"{k} {c[k]:.4f} against {r[k]:.4f} ({rise:+.4f}{', over the margin' if bad else ''})")
        lines.append(f"{candidate}: " + ", ".join(parts) + f" on {c['rows']} rows")
    return refused, lines


class ExportDecider:
    """The pace and the gate of the trainer's exports (the module's docstring). The state that a restart keeps is in
    the trainer's train_state: last_export_time, the wall-clock seconds of the last export.

    min_minutes: 0 (or None) for no pace. margin: None for no gate. max_refusals: refusals in a row before the halt.
    clock: wall-clock seconds (time.time), a fake one in the tests."""

    def __init__(self, min_minutes=0.0, margin=None, max_refusals=3, clock=time.time):
        self.min_seconds = 60.0 * (min_minutes or 0.0)
        self.margin = margin
        self.max_refusals = max_refusals
        self.clock = clock
        self.refusals = 0
        if self.min_seconds < 0:
            raise ValueError(f"-export-min-minutes must be 0 or more, not {min_minutes}")
        if margin is not None and not margin > 0:
            raise ValueError(f"-export-gate-margin must be above 0, not {margin}")
        if max_refusals < 1:
            raise ValueError(f"-export-gate-refusals must be 1 or more, not {max_refusals}")

    @property
    def gates(self):
        return self.margin is not None

    def ask(self, train_state):
        """The end of an export cycle: whether its export goes now, as the pace allows. A skip is a log line."""
        last = train_state.get("last_export_time")
        if self.min_seconds <= 0 or last is None:
            return True
        waited = self.clock() - last
        # A clock that went back counts as a long wait: the pace never stops the exports for good.
        if waited < 0 or waited >= self.min_seconds:
            return True
        logging.info(f"EXPORT PACE: skipped, the last export was {waited / 60:.1f} min ago (-export-min-minutes "
                     f"{self.min_seconds / 60:g}): training on")
        return False

    def judge(self, losses):
        """Whether the candidate exports, from the gate's losses (GateValidation.losses). Without the gate it does.
        Without validation rows the gate cannot compare, and the export goes ahead with a log line. After a refusal
        the next export comes at the next cycle's end. After max_refusals refusals in a row, it raises
        ExportGateError."""
        if not self.gates:
            return True
        if not losses:
            logging.warning("EXPORT GATE: no validation rows to compare on: the export goes ahead untested")
            return True
        refused, lines = compare(losses, self.margin)
        for line in lines:
            logging.info(f"EXPORT GATE: {line}")
        if not refused:
            logging.info(f"EXPORT GATE: export (margin {self.margin:g})")
            return True
        self.refusals += 1
        message = (f"EXPORT GATE: refused, a loss rose more than the margin {self.margin:g} against the last export: "
                   f"refusal {self.refusals} in a row of {self.max_refusals}")
        if self.refusals >= self.max_refusals:
            message += ". Halting the trainer: check its checkpoint and its rows"
            logging.error(message)
            raise ExportGateError(message)
        logging.warning(message + ", training on")
        return False

    def exported(self, train_state):
        train_state["last_export_time"] = self.clock()
        self.refusals = 0


def save_reference(path, name, step, raw_state, swa_state):
    """The last exported network's weights (raw, and SWA or None), with its name and step, for a restart's gate."""
    torch.save({"name": name, "global_step_samples": step, "model": raw_state, "swa_model": swa_state}, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_reference(path):
    """The saved reference (save_reference), or None when there is none."""
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu", weights_only=True)


def state_on_cpu(module):
    """A copy of the module's state dict on the CPU, which later training does not change."""
    return {k: v.detach().to("cpu", copy=True) for k, v in module.state_dict().items()}


def networks_on(device, make_model, states):
    """{name: network} for each name's state dict (None leaves the name out), each in a new model on the device. They
    hold the device's memory until the caller drops the dict."""
    networks = {}
    for name, state in states.items():
        if state is None:
            continue
        network = make_model()
        network.load_state_dict(state)
        networks[name] = network.to(device)
    return networks
