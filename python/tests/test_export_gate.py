"""The trainer's guards of its exports (katago/train/export_gate.py) and of fp16 training (trainloop_helpers.GnormWatcher's
grad-scale floor), with bpj5's numbers of 2026-11-01 (abscls-8mbo, abscls-ehuh, games/blueprints/docs/decisions/port.md
in the engine repository, "the trainer's export gate")."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from katago.train import export_gate, games, modelconfigs  # noqa: E402
from katago.train.data_processing_pytorch import GLOBAL_TARGETS_NC_CHANNELS, VALUE_TARGETS_NCHW_CHANNELS  # noqa: E402
from katago.train.metrics_pytorch import Metrics  # noqa: E402
from katago.train.model_pytorch import Model, policy_len  # noqa: E402
from katago.train.trainloop_helpers import GnormWatcherError, GnormWatcher  # noqa: E402

MARGIN = 0.08  # bpj5's margin: above every healthy rise of its 533 validations, below both damaged ones


class Clock:
    def __init__(self):
        self.minutes = 0.0

    def __call__(self):
        return 60.0 * self.minutes


def exports_of(decider, asks):
    """The trainer at the end of each export cycle (minutes on the clock): the cycle asks for an export, and the
    decider says whether it goes. Returns the minutes of the exports, each with the newest weights of that time."""
    state, exports = {}, []
    for minutes in asks:
        decider.clock.minutes = minutes
        if decider.ask(state) and decider.judge({}):
            decider.exported(state)
            exports.append(minutes)
    return exports


def test_a_catch_up_that_asks_every_2_6_minutes_exports_once_each_10_minutes():
    # bpj5's trainer after its restart of 09:04: epochs of 0.65 minutes, an export asked every 4 epochs (2.6 minutes).
    # Then normal running: an export asked every 23 minutes.
    catch_up = [2.6 * k for k in range(1, 11)]  # 2.6 to 26.0 minutes
    normal = [26.0 + 23.0 * k for k in range(1, 4)]  # 49.0, 72.0, 95.0
    exports = exports_of(export_gate.ExportDecider(min_minutes=10, clock=Clock()), catch_up + normal)
    # The first asked export goes at once. A cycle that ends less than 10 minutes after the last export skips its
    # export, so one export goes each 10.4 minutes, each at the end of its cycle. The export of 26.0 is skipped: the
    # last checkpoint of the catch-up still exports at its time, the end of the next cycle (49.0), and so does every
    # later one. In normal running the cycles end further apart than 10 minutes, and every asked export goes.
    assert exports == pytest.approx([2.6, 13.0, 23.4, 49.0, 72.0, 95.0])
    assert min(b - a for a, b in zip(exports, exports[1:])) >= 10


def test_tonights_catch_up_and_normal_running_with_a_pace_of_10_minutes():
    # bpj5's exports of 2026-11-01 09:04-11:00, each at the next epoch's start in train-b6c96.log (minutes after
    # 09:00). The memory guard stopped the trainer at 09:21:54, and it started again at 09:29: the checkpoint keeps the
    # time of the last export.
    asks = [10 + 10 / 60, 12 + 35 / 60, 15 + 14 / 60, 17 + 50 / 60, 20 + 32 / 60, 31 + 59 / 60, 34 + 35 / 60,
            37 + 3 / 60, 52 + 58 / 60, 76 + 23 / 60, 93 + 8 / 60, 117 + 25 / 60]
    exports = exports_of(export_gate.ExportDecider(min_minutes=10, clock=Clock()), asks)
    # 09:10:10, 09:20:32, 09:31:59, then each export of normal running at its time.
    assert exports == pytest.approx([asks[0], asks[4], asks[5]] + asks[8:])


def test_without_a_pace_every_asked_export_goes():
    asks = [2.6 * k for k in range(1, 11)]
    assert exports_of(export_gate.ExportDecider(min_minutes=0, clock=Clock()), asks) == asks


def test_a_clock_that_went_back_does_not_skip_an_export():
    clock = Clock()
    decider, state = export_gate.ExportDecider(min_minutes=10, clock=clock), {}
    clock.minutes = 100
    assert decider.ask(state)
    decider.exported(state)
    clock.minutes = 50
    assert decider.ask(state)


def losses(p0, p1, rows=3840):
    return {"p0loss": p0, "p1loss": p1, "rows": rows}


# bpj5's validations (train/b6c96/metrics_val.json): each against the validation of the last export before it.
TONIGHT = {
    # The export of 05:53 (b6c96-s30897152), the reference of the two damaged epochs.
    "s30897152": losses(2.284751, 0.823919),
    # The epoch that began at 06:08:11, and the epoch of the export b6c96-s30962304 (06:15:20), both damaged.
    "06:08": losses(2.336369, 0.913312),
    "s30962304": losses(2.714538, 1.149316),
    # The largest healthy rises: p0loss at the export b6c96-s23663488 (2026-10-30 14:26) against the export of
    # 14:03, and p1loss at 2026-11-01 10:40 against the export of 10:32.
    "s23599744": losses(2.229343, 0.821108),
    "s23663488": losses(2.299569, 0.823281),
    "s31600128": losses(2.270980, 0.798077),
    "10:40": losses(2.299508, 0.847606),
}


def gate(candidate, reference):
    return export_gate.compare({"raw": TONIGHT[candidate], "raw_reference": TONIGHT[reference]}, MARGIN)


def test_the_margin_refuses_both_damaged_validations_and_passes_the_largest_healthy_rises():
    assert gate("06:08", "s30897152")[0]  # p1loss +0.089
    assert gate("s30962304", "s30897152")[0]  # p0loss +0.430, p1loss +0.325
    assert not gate("s23663488", "s23599744")[0]  # p0loss +0.070
    assert not gate("10:40", "s31600128")[0]  # p1loss +0.050


def test_the_gate_compares_each_pair_and_a_nonfinite_loss_is_refused():
    healthy, damaged = TONIGHT["s23599744"], TONIGHT["s30962304"]
    refused, lines = export_gate.compare(
        {"raw": healthy, "raw_reference": healthy, "swa": damaged, "swa_reference": healthy}, MARGIN)
    assert refused and lines[0].startswith("raw: ") and "over the margin" in lines[1]
    nan = losses(float("nan"), 0.8)
    assert export_gate.compare({"raw": nan, "raw_reference": healthy}, MARGIN)[0]
    # A pair without its reference is left out.
    assert export_gate.compare({"raw": damaged, "swa_reference": healthy}, MARGIN) == (False, [])


def test_three_refusals_in_a_row_halt_the_trainer_and_an_export_between_them_resets_the_count():
    decider, state = export_gate.ExportDecider(margin=MARGIN, max_refusals=3), {}
    bad = {"raw": TONIGHT["s30962304"], "raw_reference": TONIGHT["s30897152"]}
    good = {"raw": TONIGHT["s30897152"], "raw_reference": TONIGHT["s30897152"]}
    for _ in range(2):
        assert decider.ask(state) and not decider.judge(bad)
    assert decider.judge(good)
    decider.exported(state)
    assert not decider.judge(bad) and not decider.judge(bad)
    with pytest.raises(export_gate.ExportGateError):
        decider.judge(bad)


def test_without_validation_rows_or_without_a_gate_the_export_goes_ahead():
    bad = {"raw": TONIGHT["s30962304"], "raw_reference": TONIGHT["s30897152"]}
    assert export_gate.ExportDecider(margin=MARGIN).judge({})
    assert export_gate.ExportDecider().judge(bad)


@pytest.mark.parametrize("kwargs", [{"min_minutes": -1}, {"margin": 0}, {"max_refusals": 0}])
def test_settings_out_of_range_are_refused(kwargs):
    with pytest.raises(ValueError):
        export_gate.ExportDecider(**kwargs)


# --- The gate's pass beside the trainer's validation, on a small network ----------------------------------------------

def network(seed):
    torch.manual_seed(seed)
    model = Model(modelconfigs.with_game(modelconfigs.config_of_name["b6c96"], "enclosure"),
                  games.get("enclosure").MAX_BOARD_LEN)
    model.initialize()
    return model


def batch_of(model, n=6):
    torch.manual_seed(0)
    size = model.pos_len
    binary = (torch.rand(n, modelconfigs.get_num_bin_input_features(model.config), size, size) < 0.2).float()
    binary[:, 0] = 1.0
    global_targets = torch.zeros(n, GLOBAL_TARGETS_NC_CHANNELS)
    global_targets[:, 0] = 1.0
    global_targets[:, [25, 26, 27, 28, 33, 34, 68, 69]] = 1.0
    score = torch.zeros(n, model.value_head.scorebelief_len)
    score[:, model.value_head.scorebelief_mid] = 100.0
    return {"binaryInputNCHW": binary,
            "globalInputNC": torch.randn(n, modelconfigs.get_num_global_input_features(model.config)),
            "policyTargetsNCMove": torch.rand(n, 2, policy_len(size, model.policy_head.policy_channels)),
            "globalTargetsNC": global_targets, "scoreDistrN": score,
            "valueTargetsNCHW": torch.randint(-1, 2, (n, VALUE_TARGETS_NCHW_CHANNELS, size, size)).float()}


def validation(model, networks, chunk_rows):
    """The gate's pass with the trainer's validation metrics, the model's Metrics."""
    metrics_obj = Metrics(1, model)

    def metrics(net, outputs, part):
        return metrics_obj.metrics_dict_batchwise(
            net, net.postprocess_output(outputs), None, part, is_training=False, soft_policy_weight_scale=8.0,
            disable_optimistic_policy=False, meta_kata_only_soft_policy=False, value_loss_scale=0.6,
            td_value_loss_scales=[0.6, 0.6, 0.6], seki_loss_scale=1.0, variance_time_loss_scale=1.0,
            main_loss_scale=None, intermediate_loss_scale=None, include_model_norms=False)

    return export_gate.GateValidation(networks, lambda net, part: net(part["binaryInputNCHW"], part["globalInputNC"]),
                                      metrics, chunk_rows), metrics


def test_the_gates_networks_give_the_same_losses_in_parts_as_on_the_whole_batch():
    model = network(1)
    batch = batch_of(model)
    whole, metrics = validation(model, {"raw": model}, chunk_rows=6)
    parts, _ = validation(model, {"raw": model}, chunk_rows=4)
    whole.evaluate(batch)
    parts.evaluate(batch)
    assert model.training  # evaluate runs it in eval mode and gives its mode back
    assert parts.losses()["raw"]["rows"] == 6
    for k in export_gate.LOSSES:
        assert parts.losses()["raw"][k] == pytest.approx(whole.losses()["raw"][k], rel=1e-5)
    # The trainer's own network adds the metrics of its validation of the whole batch: the same sums.
    added, _ = validation(model, {}, chunk_rows=4)
    with torch.no_grad():
        model.eval()
        added.add("raw", metrics(model, model(batch["binaryInputNCHW"], batch["globalInputNC"]), batch), 6)
    assert added.losses()["raw"]["p0loss"] == pytest.approx(whole.losses()["raw"]["p0loss"], rel=1e-5)


def test_a_network_against_its_own_saved_weights_passes_and_a_damaged_one_is_refused(tmp_path):
    model = network(1)
    batch = batch_of(model)
    path = str(tmp_path / export_gate.REFERENCE_FILE)
    export_gate.save_reference(path, "b6c96-s1-d1", 1, export_gate.state_on_cpu(model), None)
    saved = export_gate.load_reference(path)
    assert (saved["name"], saved["global_step_samples"], saved["swa_model"]) == ("b6c96-s1-d1", 1, None)
    make = lambda: Model(model.config, model.pos_len)
    reference = export_gate.networks_on("cpu", make, {"raw_reference": saved["model"], "swa_reference": None})
    assert list(reference) == ["raw_reference"]
    same, _ = validation(model, {"raw": model, **reference}, chunk_rows=3)
    same.evaluate(batch)
    refused, lines = export_gate.compare(same.losses(), MARGIN)
    assert not refused and "+0.0000" in lines[0]
    # Damaged weights: noise on every weight, as steps of noise gradients leave them.
    torch.manual_seed(2)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    damaged, _ = validation(model, {"raw": model, **reference}, chunk_rows=3)
    damaged.evaluate(batch)
    assert export_gate.compare(damaged.losses(), MARGIN)[0]
    assert export_gate.load_reference(str(tmp_path / "none.ckpt")) is None


def test_the_reference_is_a_copy_that_later_training_does_not_change():
    model = network(1)
    state = export_gate.state_on_cpu(model)
    name = next(iter(state))
    before = state[name].clone()
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    assert torch.equal(state[name], before)


# --- The grad-scale floor -------------------------------------------------------------------------------------------

# The scale at each epoch's start in bpj5's train-b6c96.log ("Current grad scale"). The halted run of 2026-11-01
# 05:23-06:23, from 05:24:11 to 06:23:09: below 1 from the reading of 05:53:02 (index 6), before its exports of 05:54
# (b6c96-s30897152) and 06:17 (b6c96-s30962304).
HALTED_RUN = [65536.0, 64.0, 32.0, 16.0, 4.0, 1.0, 0.5, 0.015625, 0.0078125, 0.00048828125, 3.0517578125e-05,
              9.5367431640625e-07]
# The healthy run of 2026-10-30 10:23-22:07 (144 readings), its readings each time the scale changed: its lowest, 8,
# is the lowest of every healthy run of bpj5 (16, 8, 256, 256, 128, 16, 64, 128, and 256 twice after the fix).
HEALTHY_RUN = [65536.0, 256.0, 128.0, 64.0, 32.0, 16.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 128.0, 256.0, 128.0]
FLOOR = 1.0  # bpj5's floor


def first_halt(watcher, scales):
    for i, scale in enumerate(scales):
        try:
            watcher.observe({"gnorm_batch": 600.0}, gnorm_cap=1768.0, grad_scale=scale)
        except GnormWatcherError:
            return i
    return None


def test_the_floor_halts_tonights_run_at_its_first_scale_below_1_and_no_healthy_run():
    assert first_halt(GnormWatcher(grad_scale_floor=FLOOR), HALTED_RUN) == 6
    assert first_halt(GnormWatcher(grad_scale_floor=FLOOR), HEALTHY_RUN) is None
    # The two runs with rows of target sum 0 before it (02:46-04:00 and 04:00-05:23): at 0.5 too.
    assert first_halt(GnormWatcher(grad_scale_floor=FLOOR), [65536.0, 256.0, 128.0, 64.0, 16.0, 1.0, 1.0, 0.5]) == 7
    assert first_halt(GnormWatcher(grad_scale_floor=FLOOR), [65536.0, 256.0, 64.0, 32.0, 2.0, 2.0, 1.0, 0.5]) == 7


def test_without_a_floor_or_a_scale_nothing_halts(monkeypatch):
    assert first_halt(GnormWatcher(), HALTED_RUN) is None
    watcher = GnormWatcher(grad_scale_floor=FLOOR)
    assert first_halt(watcher, [None] * 12) is None
    monkeypatch.setenv("KATAGO_GNORM_WATCHER_HALT", "0")
    assert first_halt(GnormWatcher(grad_scale_floor=FLOOR), HALTED_RUN) is None  # it warns only
    with pytest.raises(ValueError):
        GnormWatcher(grad_scale_floor=0)
