from typing import Any, Dict, List
import math

from ..train import games
from ..train import modelconfigs
from ..train.model_pytorch import EXTRA_SCORE_DISTR_RADIUS, policy_len, Model, compute_gain, ExtraOutputs, MetadataEncoder
from ..train.trainloop_helpers import env_flag

import torch
import torch.nn
import torch.nn.functional

def cross_entropy(pred_logits, target_probs, dim):
    return -torch.sum(target_probs * torch.nn.functional.log_softmax(pred_logits, dim=dim), dim=dim)

def huber_loss(x, y, delta):
    abs_diff = torch.abs(x - y)
    return torch.where(
        abs_diff > delta,
        (0.5 * delta * delta) + delta * (abs_diff - delta),
        0.5 * abs_diff * abs_diff,
    )

def constant_like(data, other_tensor):
    return torch.tensor(data, dtype=other_tensor.dtype, device=other_tensor.device, requires_grad=False)

class Metrics:
    def __init__(self, world_size: int, raw_model: Model):
        self.world_size = world_size
        self.pos_len = raw_model.pos_len
        self.pos_area = raw_model.pos_len * raw_model.pos_len
        # Enclosure fork: the game's facts (katago/train/games). One policy entry per action, plus pass.
        game = modelconfigs.get_game(raw_model.config)
        self.policy_channels = game.POLICY_CHANNELS
        self.policy_len = policy_len(raw_model.pos_len, self.policy_channels)
        self.score_scale = raw_model.score_scale
        self.time_scale = raw_model.time_scale
        self.writes_ownership = bool(game.TARGETS & games.TARGET_OWNERSHIP)
        # KataGame fork: the game's coefficients of the ownership, seki and income losses, and the points where the
        # income loss reads its target (katago/train/games: LOSS_COEFFICIENTS, income_mask).
        self.loss_coefficients = games.loss_coefficients(game)
        self.income_mask = getattr(game, "income_mask", None)
        self.value_len = 3
        self.num_td_values = 3
        self.num_futurepos_values = 2
        self.num_seki_logits = 4
        self.scorebelief_len = 2 * (self.pos_len*self.pos_len + EXTRA_SCORE_DISTR_RADIUS)

        self.scoremean_multiplier = raw_model.scoremean_multiplier

        self.score_belief_offset_vector = raw_model.value_head.score_belief_offset_vector
        # Keeping the seki moving average on the model device avoids a per-batch
        # GPU->CPU sync in the training loss and lets the loss be torch.compiled.
        self.seki_ema_on_device = env_flag("KATAGO_SEKI_EMA_ON_DEVICE", default=True)
        if self.seki_ema_on_device:
            metric_device = self.score_belief_offset_vector.device
            self.moving_unowned_proportion_sum = torch.zeros([], device=metric_device, dtype=torch.float32)
            self.moving_unowned_proportion_weight = torch.zeros([], device=metric_device, dtype=torch.float32)
        else:
            self.moving_unowned_proportion_sum = 0.0
            self.moving_unowned_proportion_weight = 0.0

    def state_dict(self):
        # Checkpoints always store plain floats regardless of where the moving
        # average lives at runtime.
        moving_sum = self.moving_unowned_proportion_sum
        moving_weight = self.moving_unowned_proportion_weight
        if isinstance(moving_sum, torch.Tensor):
            moving_sum = moving_sum.item()
        if isinstance(moving_weight, torch.Tensor):
            moving_weight = moving_weight.item()
        return dict(
            moving_unowned_proportion_sum = moving_sum,
            moving_unowned_proportion_weight = moving_weight,
        )
    def load_state_dict(self, state_dict: Dict[str,Any]):
        moving_sum = state_dict["moving_unowned_proportion_sum"]
        moving_weight = state_dict["moving_unowned_proportion_weight"]
        if isinstance(moving_sum, torch.Tensor):
            moving_sum = moving_sum.item()
        if isinstance(moving_weight, torch.Tensor):
            moving_weight = moving_weight.item()
        if self.seki_ema_on_device:
            self.moving_unowned_proportion_sum.fill_(moving_sum)
            self.moving_unowned_proportion_weight.fill_(moving_weight)
        else:
            self.moving_unowned_proportion_sum = moving_sum
            self.moving_unowned_proportion_weight = moving_weight

    def loss_policy_player_samplewise(self, pred_logits, target_probs, weight, global_weight):
        assert pred_logits.shape[1:] == (self.policy_len,)
        assert target_probs.shape == pred_logits.shape
        loss = cross_entropy(pred_logits, target_probs, dim=1)
        return global_weight * weight * loss

    def loss_policy_opponent_samplewise(self, pred_logits, target_probs, weight, global_weight):
        assert pred_logits.shape[1:] == (self.policy_len,)
        assert target_probs.shape == pred_logits.shape
        loss = cross_entropy(pred_logits, target_probs, dim=1)
        return 0.15 * global_weight * weight * loss

    def loss_qvalues_samplewise(self, pred_wl_pretanh, pred_score_prescaled, target_wl, target_score, target_visits, global_weight):
        n = pred_wl_pretanh.shape[0]
        assert pred_wl_pretanh.shape == (n, self.policy_len)
        assert pred_score_prescaled.shape == (n, self.policy_len)
        assert target_wl.shape == (n, self.policy_len)
        assert target_score.shape == (n, self.policy_len)
        assert target_visits.shape == (n, self.policy_len)

        mask = (target_visits != 0).float()
        sqrtvisits = torch.sqrt(target_visits)
        sum_sqrtvisits = torch.sum(sqrtvisits, dim=1)

        pred_wl_logits = pred_wl_pretanh * mask * 2.0
        target_wl_probs = (1.0 + target_wl) / 2.0

        loss_qvalues_winloss_by_move = torch.nn.functional.binary_cross_entropy_with_logits(pred_wl_logits, target_wl_probs, reduction="none")
        loss_qvalues_winloss = torch.sum(
            loss_qvalues_winloss_by_move * sqrtvisits,
            dim=1,
        ) / (sum_sqrtvisits + 1.0)  # Add 1.0 to sum of sqrt visits so that we don't divide by 0 if we have no such data.

        pred_score = pred_score_prescaled * mask * self.scoremean_multiplier
        loss_qvalues_score_by_move = huber_loss(pred_score, target_score, delta = 12.0 * self.score_scale) * sqrtvisits  # Enclosure fork: the game's score scale
        loss_qvalues_score = torch.sum(
            loss_qvalues_score_by_move * sqrtvisits,
            dim=1,
        ) / (sum_sqrtvisits + 1.0)  # Add 1.0 to sum of sqrt visits so that we don't divide by 0 if we have no such data.

        return 1.5 * global_weight * loss_qvalues_winloss, 0.0008 / self.score_scale ** 2 * global_weight * loss_qvalues_score


    def loss_value_samplewise(self, pred_logits, target_probs, weight, global_weight):
        n = pred_logits.shape[0]
        assert pred_logits.shape == (n, self.value_len)
        assert target_probs.shape == (n, self.value_len)
        assert weight.shape == (n,)
        loss = cross_entropy(pred_logits, target_probs, dim=1)
        return 1.20 * global_weight * weight * loss

    def loss_td_value_samplewise(self, pred_logits, target_probs, weight, global_weight):
        n = pred_logits.shape[0]
        assert pred_logits.shape == (n, self.num_td_values, self.value_len)
        assert target_probs.shape == (n, self.num_td_values, self.value_len)
        assert weight.shape == (n,)
        assert global_weight.shape == (n,)
        loss = cross_entropy(pred_logits, target_probs, dim=2) - cross_entropy(torch.log(target_probs + 1.0e-30), target_probs, dim=2)
        return 1.20 * global_weight.unsqueeze(1) * weight.unsqueeze(1) * loss

    def loss_td_score_samplewise(self, pred, target, weight, global_weight):
        n = pred.shape[0]
        assert pred.shape == (n, self.num_td_values)
        assert target.shape == (n, self.num_td_values)
        loss = torch.sum(huber_loss(pred, target, delta = 12.0 * self.score_scale), dim=1)  # Enclosure fork: the game's score scale
        return 0.0004 / self.score_scale ** 2 * global_weight * weight * loss


    def loss_ownership_samplewise(self, pred_pretanh, target, weight, mask, mask_sum_hw, global_weight):
        # This uses a formulation where each batch element cares about its average loss.
        # In particular this means that ownership loss predictions on small boards "count more" per spot.
        # Not unlike the way that policy and value loss are also equal-weighted by batch element.
        n = pred_pretanh.shape[0]
        assert pred_pretanh.shape == (n, 1, self.pos_len, self.pos_len)
        assert target.shape == (n, self.pos_len, self.pos_len)
        assert mask.shape == (n, self.pos_len, self.pos_len)
        assert mask_sum_hw.shape == (n,)
        pred_logits = pred_pretanh.view(-1,self.pos_area) * 2.0
        target_probs = (1.0 + target.view(-1,self.pos_area)) / 2.0
        loss = torch.sum(
            torch.nn.functional.binary_cross_entropy_with_logits(pred_logits, target_probs, reduction="none") * mask.view(-1,self.pos_area),
            dim=1,
        ) / mask_sum_hw
        return self.loss_coefficients["ownership"] * global_weight * weight * loss  # KataGame fork: KataGo's 1.5


    def loss_scoring_samplewise(self, pred_scoring, target, weight, mask, mask_sum_hw, global_weight):
        n = pred_scoring.shape[0]
        assert pred_scoring.shape == (n, 1, self.pos_len, self.pos_len)
        assert target.shape == (n, self.pos_len, self.pos_len)
        assert mask.shape == (n, self.pos_len, self.pos_len)
        assert mask_sum_hw.shape == (n,)

        loss = torch.sum(torch.square(pred_scoring.squeeze(1) - target) * mask, dim=(1,2)) / mask_sum_hw
        # Simple huberlike transform to reduce crazy values
        loss = 4.0 * (torch.sqrt(loss * 0.5 + 1.0) - 1.0)
        return global_weight * weight * loss


    def loss_futurepos_samplewise(self, pred_pretanh, target, weight, mask, mask_sum_hw, global_weight):
        # The futurepos targets extrapolate a fixed number of steps into the future independent
        # of board size. So unlike the ownership above, generally a fixed number of spots are going to be
        # "wrong" independent of board size, so we should just equal-weight the prediction per spot.
        # However, on larger boards often the entropy of where the future moves will be should be greater
        # and also in the event of capture, there may be large captures that don't occur on small boards,
        # causing some scaling with board size. So, I dunno, let's compromise and scale by sqrt(boardarea).
        # Also, the further out targets should be weighted a little less due to them being higher entropy
        # due to simply being farther in the future, so multiply by [1,0.25].
        n = pred_pretanh.shape[0]
        assert pred_pretanh.shape == (n, self.num_futurepos_values, self.pos_len, self.pos_len)
        assert target.shape == (n, self.num_futurepos_values, self.pos_len, self.pos_len)
        assert mask.shape == (n, self.pos_len, self.pos_len)
        assert mask_sum_hw.shape == (n,)
        loss = torch.square(torch.tanh(pred_pretanh) - target) * mask.unsqueeze(1)
        loss = loss * constant_like([1.0,0.25], loss).view(1,2,1,1)
        loss = torch.sum(loss, dim=(1, 2, 3)) / torch.sqrt(mask_sum_hw)
        return 0.25 * global_weight * weight * loss


    def loss_regret_value_samplewise(self, pred_regret, target_regret, weight, global_weight):
        # Enclosure fork (owner, 2026-10-04): RGSC's regret value head, squared error against Eq. 2's regret (the
        # authors' code: mse_loss against the label, in [0, 4]).
        return global_weight * weight * torch.square(pred_regret - target_regret)

    def loss_regret_ranking(self, pred_ranking, target_regret, weight, global_weight, group_size: int = 32):
        # Enclosure fork (owner, 2026-10-04): RGSC's ranking loss as the authors' code computes it
        # (minizero/learner/train.py): rows sorted by regret, dealt by stride into groups, each group's loss
        # -logsumexp(log_softmax(ranking) + w * regret) with w the group's min-max normalised regret squared; the
        # mean over groups. Groups of 32 (owner: the code fixes 32 groups, which at batch 256 holds 8 each). Rows
        # without a regret target take no part. Returned as that mean times the rows taking part, so that, like the
        # other losses (sums over rows, divided by the weight later), it weighs as the authors' mean does.
        rows = torch.nonzero((weight > 0) & (global_weight > 0), as_tuple=False).view(-1)
        n = int(rows.shape[0])
        if n < 2:
            return pred_ranking.sum() * 0.0
        num_groups = max(1, n // group_size)
        regret = target_regret[rows]
        ranking = pred_ranking[rows]
        order = torch.argsort(regret, descending=True)
        total = pred_ranking.new_zeros(())
        for i in range(num_groups):
            idx = order[i::num_groups]
            r = regret[idx]
            log_p = torch.log_softmax(ranking[idx], dim=0)
            r_norm = (r - r.min()) / (r.max() - r.min() + 1e-8)
            total = total - torch.logsumexp(log_p + r_norm.pow(2) * r, dim=0)
        return total / num_groups * n

    def loss_income_samplewise(self, pred_logits, target, weight, mask, mask_sum_hw, global_weight, income_mask=None):
        # Enclosure fork (owner, 2026-10-03; iviazovetskyi's ownership target): each player's territory income per
        # point (its enclosed area in the unit square around the point, averaged over the remaining turn ends; the
        # mover's, then the opponent's), as binary cross-entropy, averaged over the board's points and both planes,
        # weighted as KataGo's ownership (1.5; iviazovetskyi's weight too).
        # KataGame fork: with the game's income_mask (n, 2, y, x), averaged over each channel's points, then over the
        # channels that have points (Blueprints' cells that can go), and with the game's coefficient.
        n = pred_logits.shape[0]
        assert pred_logits.shape == (n, 2, self.pos_len, self.pos_len)
        assert target.shape == (n, 2, self.pos_len, self.pos_len)
        assert mask.shape == (n, self.pos_len, self.pos_len)
        assert mask_sum_hw.shape == (n,)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_logits, target, reduction="none") * mask.unsqueeze(1)
        if income_mask is None:
            loss = torch.sum(loss, dim=(1, 2, 3)) / (2.0 * mask_sum_hw)
        else:
            assert income_mask.shape == (n, 2, self.pos_len, self.pos_len)
            points = torch.sum(income_mask, dim=(2, 3))
            per_channel = torch.sum(loss * income_mask, dim=(2, 3)) / torch.clamp(points, min=1.0)
            loss = torch.sum(per_channel, dim=1) / torch.clamp(torch.sum((points > 0).to(loss.dtype), dim=1), min=1.0)
        return self.loss_coefficients["income"] * global_weight * weight * loss


    def loss_edgecut_samplewise(self, pred_logits, target, weight, mask, mask_sum_hw, global_weight):
        # Enclosure fork (owner, 2026-10-03): edges that get cut, in the seki head's three sign logits. Per point
        # on an edge, cross-entropy over: not cut (0), the mover's cut (+1), the opponent's cut (-1), averaged over
        # those points (2 marks points on no edge, where the question does not exist: averaging over the whole
        # board would dilute it by the empty share), with a plain weight of 1: KataGo's adaptive seki scale and its
        # "neutral" logit track how rare seki and unowned points are in Go and do not carry over (ENCLOSURE.md,
        # "Spatial targets", 2).
        # KataGame fork: with the game's coefficient (Blueprints' cells captured before the game's end, class 2 on
        # the points without a cell that a capture can take).
        n = pred_logits.shape[0]
        assert pred_logits.shape == (n, self.num_seki_logits, self.pos_len, self.pos_len)
        assert target.shape == (n, self.pos_len, self.pos_len)
        on_edge = (target != 2).to(pred_logits.dtype) * mask
        t = target * on_edge
        sign_target = torch.stack((1.0 - torch.square(t), torch.nn.functional.relu(t), torch.nn.functional.relu(-t)), dim=1)
        loss = torch.sum(cross_entropy(pred_logits[:, 0:3, :, :], sign_target, dim=1) * on_edge, dim=(1, 2))
        loss = loss / torch.clamp(torch.sum(on_edge, dim=(1, 2)), min=1.0)
        return self.loss_coefficients["seki"] * global_weight * weight * loss

    def loss_seki_samplewise(self, pred_logits, target, target_ownership, weight, mask, mask_sum_hw, global_weight, is_training, skip_moving_update):
        assert self.num_seki_logits == 4
        n = pred_logits.shape[0]
        assert pred_logits.shape == (n, self.num_seki_logits, self.pos_len, self.pos_len)
        assert target.shape == (n, self.pos_len, self.pos_len)
        assert target_ownership.shape == (n, self.pos_len, self.pos_len)
        assert mask.shape == (n, self.pos_len, self.pos_len)
        assert mask_sum_hw.shape == (n,)

        owned_target = torch.square(target_ownership)
        unowned_target = 1.0 - owned_target
        unowned_proportion = torch.sum(unowned_target * mask, dim=(1, 2)) / (1.0 + mask_sum_hw)
        unowned_proportion = torch.mean(unowned_proportion * weight)
        if is_training:
            if not skip_moving_update:
                if self.seki_ema_on_device:
                    with torch.no_grad():
                        self.moving_unowned_proportion_sum.mul_(0.998).add_(unowned_proportion.detach())
                        self.moving_unowned_proportion_weight.mul_(0.998).add_(1.0)
                else:
                    self.moving_unowned_proportion_sum *= 0.998
                    self.moving_unowned_proportion_weight *= 0.998
                    self.moving_unowned_proportion_sum += unowned_proportion.item()
                    self.moving_unowned_proportion_weight += 1.0
            moving_unowned_proportion = self.moving_unowned_proportion_sum / self.moving_unowned_proportion_weight
            seki_weight_scale = 8.0 * 0.005 / (0.005 + moving_unowned_proportion)
        else:
            seki_weight_scale = 7.0

        # Loss for predicting the exact sign of seki points
        sign_pred = pred_logits[:, 0:3, :, :]
        sign_target = torch.stack(
            (
                1.0 - torch.square(target),
                torch.nn.functional.relu(target),
                torch.nn.functional.relu(-target),
            ),
            dim=1,
        )
        loss_sign = torch.sum(cross_entropy(sign_pred, sign_target, dim=1) * mask, dim=(1, 2))

        # Loss for generally predicting points that nobody will own
        neutral_pred = torch.stack(
            (pred_logits[:, 3, :, :], torch.zeros_like(target_ownership)), dim=1
        )
        neutral_target = torch.stack((unowned_target, owned_target), dim=1)
        loss_neutral = torch.sum(cross_entropy(neutral_pred, neutral_target, dim=1) * mask, dim=(1, 2))

        loss = loss_sign + 0.5 * loss_neutral
        loss = loss / mask_sum_hw
        return (global_weight * seki_weight_scale * weight * loss, seki_weight_scale)


    def loss_scoremean_samplewise(self, pred, target, weight, global_weight):
        # Huber will incentivize this to not actually converge to the mean,
        #but rather something meanlike locally and something medianlike
        # for very large possible losses. This seems... okay - it might actually
        # be what users want.
        assert pred.shape == target.shape
        assert pred.ndim == 1
        loss = huber_loss(pred, target, delta = 12.0 * self.score_scale)  # Enclosure fork: the game's score scale
        return 0.0015 / self.score_scale ** 2 * global_weight * weight * loss


    def loss_scorebelief_cdf_samplewise(self, pred_logits, target_probs, weight, global_weight):
        assert pred_logits.shape[1:] == (self.scorebelief_len,)
        assert target_probs.shape == pred_logits.shape
        pred_cdf = torch.cumsum(torch.nn.functional.softmax(pred_logits, dim=1), dim=1)
        target_cdf = torch.cumsum(target_probs, dim=1)
        loss = torch.sum(torch.square(pred_cdf-target_cdf),axis=1)
        return 0.020 * global_weight * weight * loss

    def loss_scorebelief_pdf_samplewise(self, pred_logits, target_probs, weight, global_weight):
        assert pred_logits.shape[1:] == (self.scorebelief_len,)
        assert target_probs.shape == pred_logits.shape
        loss = cross_entropy(pred_logits, target_probs, dim=1)
        return 0.020 * global_weight * weight * loss

    def loss_scorestdev_samplewise(self, pred, scorebelief_logits, global_weight):
        assert pred.ndim == 1
        assert scorebelief_logits.shape[1:] == (self.scorebelief_len,)
        assert self.score_belief_offset_vector.shape == (self.scorebelief_len,)
        scorebelief_probs = torch.nn.functional.softmax(scorebelief_logits, dim=1)
        expected_score_from_belief = torch.sum(scorebelief_probs * self.score_belief_offset_vector.view(1,-1),dim=1,keepdim=True)
        stdev_of_belief = torch.sqrt(0.001 + torch.sum(
            scorebelief_probs * torch.square(
                self.score_belief_offset_vector.view(1,-1) - expected_score_from_belief
            ),
            dim=1
        ))
        loss = huber_loss(pred, stdev_of_belief, delta = 10.0 * self.score_scale)  # Enclosure fork: the game's score scale
        return 0.001 / self.score_scale ** 2 * global_weight * loss

    def loss_lead_samplewise(self, pred, target, weight, global_weight):
        # Huber will incentivize this to not actually converge to the mean,
        #but rather something meanlike locally and something medianlike
        # for very large possible losses. This seems... okay - it might actually
        # be what users want.
        assert pred.shape == target.shape
        assert pred.ndim == 1
        loss = huber_loss(pred, target, delta = 8.0 * self.score_scale)  # Enclosure fork: the game's score scale
        return 0.0060 / self.score_scale ** 2 * global_weight * weight * loss

    def loss_variance_time_samplewise(self, pred, target, weight, global_weight):
        assert pred.shape == target.shape
        assert pred.ndim == 1
        # Even if the training target is 0, add a tiny bit of irreducible error for regularizing the prediction.
        loss = huber_loss(pred, target + 1.0e-5, delta = 50.0 * self.time_scale)  # Enclosure fork: moves, the game's time scale
        return 0.0003 / self.time_scale ** 2 * global_weight * weight * loss


    def loss_shortterm_value_error_samplewise(self, pred, td_value_pred_logits, td_value_target_probs, weight, global_weight):
        td_value_pred_probs = torch.softmax(td_value_pred_logits[:,2,:],axis=1)
        predvalue = (td_value_pred_probs[:,0] - td_value_pred_probs[:,1]).detach()
        realvalue = td_value_target_probs[:,2,0] - td_value_target_probs[:,2,1]
        # Even if the training target is 0, add a tiny bit of irreducible error for regularizing the prediction, 0.01%.
        sqerror = torch.square(predvalue-realvalue) + 1.0e-8
        loss = huber_loss(pred, sqerror, delta = 0.4)
        return 2.0 * global_weight * weight * loss

    def loss_shortterm_score_error_samplewise(self, pred, td_score_pred, td_score_target, weight, global_weight):
        predscore = td_score_pred[:,2].detach()
        realscore = td_score_target[:,2]
        # Even if the training target is 0, add a tiny bit of irreducible error for regularizing the prediction, one hundredth of a point.
        sqerror = torch.square(predscore-realscore) + 1.0e-4 * self.score_scale ** 2  # Enclosure fork: the game's score scale
        loss = huber_loss(pred, sqerror, delta = 100.0 * self.score_scale ** 2)
        return 0.00002 / self.score_scale ** 4 * global_weight * weight * loss

    def accuracy1(self, pred_logits, target_probs, weight, global_weight):
        return torch.sum(global_weight * weight * (torch.argmax(pred_logits,dim=1) == torch.argmax(target_probs,dim=1)))

    def target_entropy(self, target_probs, weight, global_weight):
        return torch.sum(global_weight * weight * -torch.sum(target_probs * torch.log(target_probs + 1e-30), dim=-1))

    def square_value(self, value_logits, global_weight):
        return torch.sum(global_weight * torch.square(torch.sum(torch.softmax(value_logits,dim=1) * constant_like([1,-1,0],global_weight), dim=1)))

    @staticmethod
    def get_model_norms(raw_model):
        reg_dict: Dict[str,List] = {}
        raw_model.add_reg_dict(reg_dict)

        device = reg_dict["normal"][0].device
        dtype = torch.float32

        with torch.no_grad():
            norms: Dict[str,float] = {}
            for group_name in reg_dict:
                if len(reg_dict[group_name]) > 0:
                    norm = torch.zeros([],device=device,dtype=dtype)
                    for tensor in reg_dict[group_name]:
                        norm += torch.sum(tensor * tensor)
                    norms[group_name] = torch.sqrt(norm).detach().cpu().item()

        return norms

    @staticmethod
    def get_model_norm_metrics(raw_model):
        """Model norm metrics in the naming used by the training metrics dict."""
        norms = Metrics.get_model_norms(raw_model)
        return {f"norm_{group_name}_batch": value for group_name, value in norms.items()}

    # A channel counts as dead when its norm is at most this fraction of the reference norm of
    # its tensor, see reference_channel_norm. Shared with resurrect_dead_channels.py.
    DEAD_CHANNEL_NORM_FRAC = 0.01

    @staticmethod
    def is_output_channel_tensor(param):
        """True for nn.Linear [out, in] and conv [out, in, kh, kw] weights, whose dim 0 indexes
        output channels. Biases and norm gammas and betas are excluded, including ones stored as
        [1, C, 1, 1]."""
        return param.dim() in (2, 4) and param.shape[0] >= 2 and param[0].numel() >= 2

    @staticmethod
    def reference_channel_norm(norms):
        """Typical norm of the live channels of a tensor, given the norms of all its channels:
        the median over channels whose norm exceeds 0.1% of the largest, so that a tensor where
        most channels are exactly zero still gets a reference from the live ones. Returns a 0-dim
        tensor, which is 0 when every channel is near zero."""
        largest = norms.max() if norms.numel() > 0 else torch.zeros([], device=norms.device)
        live = norms[norms > 1e-3 * largest]
        return live.median() if live.numel() > 0 else torch.zeros([], device=norms.device)

    @staticmethod
    def get_output_channel_metrics(raw_model, floor_norms_by_param=None):
        """Fractions of output channels, over all output-channel tensors of the model, that are
        dead or sitting at the weight decay floor.

        deadrows_batch: channels whose weight norm is at most DEAD_CHANNEL_NORM_FRAC times the
        tensor's reference channel norm. Exactly-zero channels always count. Such a channel
        contributes nothing and, if the tensors reading it are also zero, can never recover.
        floorrows_batch: channels whose norm is at or below the tensor's weight decay floor. Only
        reported when floors are given (see -wd-floor-frac in train.py)."""
        num_channels = 0
        num_dead = 0
        num_at_floor = 0
        with torch.no_grad():
            for param in raw_model.parameters():
                if not Metrics.is_output_channel_tensor(param):
                    continue
                r = torch.linalg.vector_norm(param, dim=tuple(range(1, param.dim())), dtype=torch.float32)
                ref = Metrics.reference_channel_norm(r)
                num_channels += r.numel()
                num_dead += int((r <= Metrics.DEAD_CHANNEL_NORM_FRAC * ref).sum().item())
                if floor_norms_by_param is not None and param in floor_norms_by_param:
                    num_at_floor += int((r <= floor_norms_by_param[param]).sum().item())
        metrics = {"deadrows_batch": num_dead / max(1, num_channels)}
        if floor_norms_by_param is not None:
            metrics["floorrows_batch"] = num_at_floor / max(1, num_channels)
        return metrics

    def get_specific_norms_and_gradient_stats(self,raw_model):
        with torch.no_grad():
            params = {}
            for name, param in raw_model.named_parameters():
                params[name] = param

            stats = {}
            def add_norm_and_grad_stats(name):
                param = params[name]
                if name.endswith(".weight"):
                    fanin = param.shape[1]
                elif name.endswith(".gamma"):
                    fanin = 1
                elif name.endwith(".beta"):
                    fanin = 1
                else:
                    assert False, "unimplemented case to compute stats on parameter"

                # 1.0 means that the average squared magnitude of a parameter in this tensor is around where
                # it would be at initialization, assuming it uses the activation that the model generally
                # uses (e.g. relu or mish)
                param_scale = torch.sqrt(torch.mean(torch.square(param))) / compute_gain(raw_model.activation) * math.sqrt(fanin)
                stats[f"{name}.SCALE_batch"] = param_scale

                # How large is the gradient, on the same scale?
                stats[f"{name}.GRADSC_batch"] = torch.sqrt(torch.mean(torch.square(param.grad))) / compute_gain(raw_model.activation) * math.sqrt(fanin)

                # And how large is the component of the gradient that is orthogonal to the overall magnitude of the parameters?
                orthograd = param.grad - param * (torch.sum(param.grad * param) / (1e-20 + torch.sum(torch.square(param))))
                stats[f"{name}.OGRADSC_batch"] = torch.sqrt(torch.mean(torch.square(orthograd))) / compute_gain(raw_model.activation) * math.sqrt(fanin)

            add_norm_and_grad_stats("blocks.1.normactconvp.conv.weight")
            add_norm_and_grad_stats("blocks.1.blockstack.0.normactconv1.conv.weight")
            add_norm_and_grad_stats("blocks.1.blockstack.0.normactconv2.conv.weight")
            add_norm_and_grad_stats("blocks.1.blockstack.1.normactconv2.norm.gamma")
            add_norm_and_grad_stats("blocks.1.normactconvq.conv.weight")
            add_norm_and_grad_stats("blocks.1.normactconvq.norm.gamma")

            add_norm_and_grad_stats("blocks.6.normactconvp.conv.weight")
            add_norm_and_grad_stats("blocks.6.blockstack.0.normactconv1.conv.weight")
            add_norm_and_grad_stats("blocks.6.blockstack.0.normactconv2.conv.weight")
            add_norm_and_grad_stats("blocks.6.blockstack.1.normactconv2.norm.gamma")
            add_norm_and_grad_stats("blocks.6.normactconvq.conv.weight")
            add_norm_and_grad_stats("blocks.6.normactconvq.norm.gamma")

            add_norm_and_grad_stats("blocks.10.normactconvp.conv.weight")
            add_norm_and_grad_stats("blocks.10.blockstack.0.normactconv1.conv.weight")
            add_norm_and_grad_stats("blocks.10.blockstack.0.normactconv2.conv.weight")
            add_norm_and_grad_stats("blocks.10.blockstack.1.normactconv2.norm.gamma")
            add_norm_and_grad_stats("blocks.10.normactconvq.conv.weight")
            add_norm_and_grad_stats("blocks.10.normactconvq.norm.gamma")

            add_norm_and_grad_stats("blocks.16.normactconvp.conv.weight")
            add_norm_and_grad_stats("blocks.16.blockstack.0.normactconv1.conv.weight")
            add_norm_and_grad_stats("blocks.16.blockstack.0.normactconv2.conv.weight")
            add_norm_and_grad_stats("blocks.16.blockstack.1.normactconv2.norm.gamma")
            add_norm_and_grad_stats("blocks.16.normactconvq.conv.weight")
            add_norm_and_grad_stats("blocks.16.normactconvq.norm.gamma")

            add_norm_and_grad_stats("policy_head.conv1p.weight")
            add_norm_and_grad_stats("value_head.conv1.weight")
            add_norm_and_grad_stats("intermediate_policy_head.conv1p.weight")
            add_norm_and_grad_stats("intermediate_value_head.conv1.weight")

        return stats

    def metrics_dict_batchwise(
        self,
        raw_model,
        model_output_postprocessed_byheads,
        extra_outputs,
        batch,
        is_training,
        soft_policy_weight_scale,
        disable_optimistic_policy,
        meta_kata_only_soft_policy,
        value_loss_scale,
        td_value_loss_scales,
        seki_loss_scale,
        variance_time_loss_scale,
        main_loss_scale,
        intermediate_loss_scale,
        include_model_norms=True,
    ):
        results = self.metrics_dict_batchwise_single_heads_output(
            raw_model,
            model_output_postprocessed_byheads[0],
            batch,
            is_training=is_training,
            soft_policy_weight_scale=soft_policy_weight_scale,
            disable_optimistic_policy=disable_optimistic_policy,
            meta_kata_only_soft_policy=meta_kata_only_soft_policy,
            value_loss_scale=value_loss_scale,
            td_value_loss_scales=td_value_loss_scales,
            seki_loss_scale=seki_loss_scale,
            variance_time_loss_scale=variance_time_loss_scale,
            is_intermediate=False,
            include_model_norms=include_model_norms,
        )
        if main_loss_scale is not None:
            results["loss_sum"] = main_loss_scale * results["loss_sum"]

        if raw_model.get_has_intermediate_head():
            assert len(model_output_postprocessed_byheads) > 1
            if raw_model.training:
                assert intermediate_loss_scale is not None
            else:
                if intermediate_loss_scale is None:
                    intermediate_loss_scale = 1.0

            if intermediate_loss_scale is not None:
                iresults = self.metrics_dict_batchwise_single_heads_output(
                    raw_model,
                    model_output_postprocessed_byheads[1],
                    batch,
                    is_training=is_training,
                    soft_policy_weight_scale=soft_policy_weight_scale,
                    disable_optimistic_policy=disable_optimistic_policy,
                    meta_kata_only_soft_policy=meta_kata_only_soft_policy,
                    value_loss_scale=value_loss_scale,
                    td_value_loss_scales=td_value_loss_scales,
                    seki_loss_scale=seki_loss_scale,
                    variance_time_loss_scale=variance_time_loss_scale,
                    is_intermediate=True,
                )
                for key,value in iresults.items():
                    if key != "loss_sum":
                        results["I"+key] = value
                results["loss_sum"] = results["loss_sum"] + intermediate_loss_scale * iresults["loss_sum"]

        # Only the aggregate loss participates in backward. Keeping every
        # logging component attached makes autograd treat dozens of metrics
        # as differentiable outputs and retain their graphs until logging.
        for key, value in results.items():
            if key != "loss_sum" and isinstance(value, torch.Tensor):
                results[key] = value.detach()
        return results

    def metrics_dict_batchwise_single_heads_output(
        self,
        raw_model,
        model_output_postprocessed,
        batch,
        is_training,
        soft_policy_weight_scale,
        disable_optimistic_policy,
        meta_kata_only_soft_policy,
        value_loss_scale,
        td_value_loss_scales,
        seki_loss_scale,
        variance_time_loss_scale,
        is_intermediate,
        include_model_norms=True,
    ):
        (
            policy_logits,
            value_logits,
            td_value_logits,
            pred_td_score,
            ownership_pretanh,
            pred_scoring,
            futurepos_pretanh,
            seki_logits,
            pred_scoremean,
            pred_scorestdev,
            pred_lead,
            pred_variance_time,
            pred_shortterm_value_error,
            pred_shortterm_score_error,
            scorebelief_logits,
            pred_regret,
            pred_ranking,
            pred_swing,
            pred_swing_ranking,
        ) = model_output_postprocessed

        input_binary_nchw = batch["binaryInputNCHW"]
        input_global_nc = batch["globalInputNC"]
        target_policy_ncmove = batch["policyTargetsNCMove"]
        target_global_nc = batch["globalTargetsNC"]
        score_distribution_ns = batch["scoreDistrN"]
        target_value_nchw = batch["valueTargetsNCHW"]

        mask = input_binary_nchw[:, 0, :, :].contiguous()
        mask_sum_hw = torch.sum(mask,dim=(1,2))

        n = input_binary_nchw.shape[0]
        h = input_binary_nchw.shape[2]
        w = input_binary_nchw.shape[3]

        # Enclosure fork: a move is on the board when its point is; each point has the game's policy channels.
        policymask = torch.cat((mask.view(n,h*w).repeat_interleave(self.policy_channels, dim=1),mask.new_ones((n,1))),dim=1)

        target_policy_player = target_policy_ncmove[:, 0, :]
        target_policy_player = target_policy_player / torch.sum(target_policy_player, dim=1, keepdim=True)
        target_policy_opponent = target_policy_ncmove[:, 1, :]
        target_policy_opponent = target_policy_opponent / torch.sum(target_policy_opponent, dim=1, keepdim=True)
        target_policy_player_soft = (target_policy_player + 1e-7) * policymask
        target_policy_player_soft = torch.pow(target_policy_player_soft, 0.25)
        target_policy_player_soft /= torch.sum(target_policy_player_soft, dim=1, keepdim=True)
        target_policy_opponent_soft = (target_policy_opponent + 1e-7) * policymask
        target_policy_opponent_soft = torch.pow(target_policy_opponent_soft, 0.25)
        target_policy_opponent_soft /= torch.sum(target_policy_opponent_soft, dim=1, keepdim=True)

        target_weight_policy_player = target_global_nc[:, 26]
        target_weight_policy_opponent = target_global_nc[:, 28]

        target_value = target_global_nc[:, 0:3]
        target_scoremean = target_global_nc[:, 3]
        target_td_value = torch.stack(
            (target_global_nc[:, 4:7], target_global_nc[:, 8:11], target_global_nc[:, 12:15]), dim=1
        )
        target_td_score = torch.cat(
            (target_global_nc[:, 7:8], target_global_nc[:, 11:12], target_global_nc[:, 15:16]), dim=1
        )
        target_lead = target_global_nc[:, 21]
        target_variance_time = target_global_nc[:, 22]
        global_weight = target_global_nc[:, 25]
        target_weight_ownership = target_global_nc[:, 27]
        target_weight_lead = target_global_nc[:, 29]
        target_weight_value = 1.0 - target_global_nc[:, 35]
        target_weight_td_value = 1.0 - target_global_nc[:, 24]
        target_weight_futurepos = target_global_nc[:, 33]
        target_weight_scoring = target_global_nc[:, 34]
        # Enclosure fork: the rows give a spatial target that the game does not write the weight 0, by its weight or
        # its flag (C68 for the edges that get cut in the seki slot, C69 for the income), but for C0, which C27 weighs
        # with the score's targets. So C0 trains only in a game that writes it (katago/train/games, TARGETS).
        target_weight_ownership_slot = (
            target_weight_ownership if self.writes_ownership else torch.zeros_like(target_weight_ownership)
        )
        target_weight_seki = target_weight_ownership * target_global_nc[:, 68]
        target_weight_income = target_weight_ownership * target_global_nc[:, 69]

        target_score_distribution = score_distribution_ns / 100.0

        # Enclosure fork: C0 holds a value in [-1, 1] stored x 120, as the scoring target (C++ TargetValues):
        # Enclosure's future occupancy.
        target_ownership = target_value_nchw[:, 0, :, :] / 120.0
        target_seki = target_value_nchw[:, 1, :, :]
        target_futurepos = target_value_nchw[:, 2:4, :, :]
        target_scoring = target_value_nchw[:, 4, :, :] / 120.0
        # Enclosure fork (owner, 2026-10-03): the territory income in C5-6 (x 120), on rows flagged in C69; its
        # logits are the future-position head's channels 2-3 (model_pytorch).
        target_income = target_value_nchw[:, 5:7, :, :] / 120.0
        income_logits = futurepos_pretanh[:, 2:4]
        futurepos_pretanh = futurepos_pretanh[:, 0:2]

        predict_q_values = False
        if raw_model.config["version"] <= 11:
            assert raw_model.policy_head.num_policy_outputs == 4
            policy_opt_loss_scale = 1.000
            long_policy_opt_loss_scale = 0.0
            short_policy_opt_loss_scale = 0.0
        elif raw_model.config["version"] <= 15:
            assert raw_model.policy_head.num_policy_outputs == 6
            policy_opt_loss_scale = 0.930
            long_policy_opt_loss_scale = 0.100
            short_policy_opt_loss_scale = 0.200
        elif raw_model.config["version"] <= 16:  # version 16 has predict_q_values implied
            assert raw_model.policy_head.num_policy_outputs == 8
            policy_opt_loss_scale = 0.930
            long_policy_opt_loss_scale = 0.100
            short_policy_opt_loss_scale = 0.200
            predict_q_values = True
        elif raw_model.config["version"] <= 17:
            assert raw_model.policy_head.num_policy_outputs == 6 or raw_model.policy_head.num_policy_outputs == 8
            policy_opt_loss_scale = 0.930
            long_policy_opt_loss_scale = 0.100
            short_policy_opt_loss_scale = 0.200
            predict_q_values = bool(raw_model.config.get("predict_q_values"))
        else:
            raise RuntimeError("unsupported version: " + str(raw_model.config["version"]))

        loss_policy_player = self.loss_policy_player_samplewise(
            policy_logits[:, 0, :],
            target_policy_player,
            target_weight_policy_player,
            global_weight,
        ).sum()
        loss_policy_opponent = self.loss_policy_opponent_samplewise(
            policy_logits[:, 1, :],
            target_policy_opponent,
            target_weight_policy_opponent,
            global_weight,
        ).sum()

        target_weight_policy_player_soft = target_weight_policy_player
        target_weight_policy_opponent_soft = target_weight_policy_opponent
        if meta_kata_only_soft_policy:
            metadata_input_nc = batch["metadataInputNC"]
            assert metadata_input_nc.shape[0] == target_weight_policy_player_soft.shape[0]
            # 151 indicates source 0 = katago
            target_weight_policy_player_soft = target_weight_policy_player_soft * metadata_input_nc[:,151]
            target_weight_policy_opponent_soft = target_weight_policy_opponent_soft * metadata_input_nc[:,151]

        loss_policy_player_soft = self.loss_policy_player_samplewise(
            policy_logits[:, 2, :],
            target_policy_player_soft,
            target_weight_policy_player_soft,
            global_weight,
        ).sum()
        loss_policy_opponent_soft = self.loss_policy_opponent_samplewise(
            policy_logits[:, 3, :],
            target_policy_opponent_soft,
            target_weight_policy_opponent_soft,
            global_weight,
        ).sum()

        if raw_model.config["version"] <= 11:
            target_weight_longoptimistic_policy = torch.zeros_like(global_weight)
            loss_longoptimistic_policy = torch.zeros_like(loss_policy_player)
        elif disable_optimistic_policy:
            target_weight_longoptimistic_policy = target_weight_policy_player * 0.5
            loss_longoptimistic_policy = self.loss_policy_player_samplewise(
                policy_logits[:, 4, :],
                target_policy_player,
                target_weight_longoptimistic_policy,
                global_weight,
            ).sum()
        else:
            # Long-term optimistic policy. Weight policy by:
            # Final game win squared (squaring discourages draws)
            win_squared = torch.square(
                target_global_nc[:, 0] # win (or draw, weighted by draw utility)
                + 0.5 * target_global_nc[:, 2] # noresult
            )
            # Or the score outcome of the game being around 1.5 sigma more than expected
            # Add a small amount to the variance to avoid division by zero or overly small numbers
            longterm_score_stdevs_excess = (target_global_nc[:, 3] - pred_scoremean.detach()) / torch.sqrt(torch.square(pred_scorestdev.detach()) + 0.25)
            target_weight_longoptimistic_policy = torch.clamp(
                win_squared + torch.sigmoid((longterm_score_stdevs_excess - 1.5) * 3.0),
                min=0.0,
                max=1.0,
            )
            target_weight_longoptimistic_policy = (
                target_weight_longoptimistic_policy
                * target_weight_policy_player # game has normal target
                * target_weight_ownership # and also actually ended in full ownership and score, not a sidepos
            )
            loss_longoptimistic_policy = self.loss_policy_player_samplewise(
                policy_logits[:, 4, :],
                target_policy_player,
                target_weight_longoptimistic_policy,
                global_weight,
            ).sum()

        assert len(loss_longoptimistic_policy.shape) == 0
        assert len(target_weight_longoptimistic_policy.shape) == 1
        assert target_weight_longoptimistic_policy.shape[0] == n
        target_weight_longoptimistic_policy_sum = (global_weight * target_weight_longoptimistic_policy).sum()

        if raw_model.config["version"] <= 11:
            target_weight_shortoptimistic_policy = torch.zeros_like(global_weight)
            loss_shortoptimistic_policy = torch.zeros_like(loss_policy_player)
        elif disable_optimistic_policy:
            target_weight_shortoptimistic_policy = target_weight_policy_player * 0.5
            loss_shortoptimistic_policy = self.loss_policy_player_samplewise(
                policy_logits[:, 5, :],
                target_policy_player,
                target_weight_shortoptimistic_policy,
                global_weight,
            ).sum()
        else:
            # Short-term optimistic policy. Weight policy by:
            # The shortterm value outcome being around 1.5 sigma more than expected
            # Add a small amount to the variance to avoid division by zero or overly small numbers
            shortterm_value_actual = target_global_nc[:, 12] - target_global_nc[:, 13]
            shortterm_value_pred = torch.nn.functional.softmax(td_value_logits[:, 2, :].detach(), dim=1)
            shortterm_value_pred = shortterm_value_pred[:, 0] - shortterm_value_pred[:, 1]
            shortterm_value_stdevs_excess = (shortterm_value_actual - shortterm_value_pred) / torch.sqrt(pred_shortterm_value_error.detach() + 0.0001)
            # Or the shortterm score outcome being around 1.5 sigma more than expected
            # Add a small amount to the variance to avoid division by zero or overly small numbers
            shortterm_score_stdevs_excess = (target_global_nc[:, 15] - pred_td_score[:,2].detach()) / torch.sqrt(pred_shortterm_score_error.detach() + 0.25)
            target_weight_shortoptimistic_policy = torch.clamp(
                torch.sigmoid((shortterm_value_stdevs_excess - 1.5) * 3.0) + torch.sigmoid((shortterm_score_stdevs_excess - 1.5) * 3.0),
                min=0.0,
                max=1.0,
            )
            target_weight_shortoptimistic_policy = (
                target_weight_shortoptimistic_policy
                * target_weight_policy_player # game has normal target
                * target_weight_ownership # and also actually ended in full ownership and score, not a sidepos
            )
            loss_shortoptimistic_policy = self.loss_policy_player_samplewise(
                policy_logits[:, 5, :],
                target_policy_player,
                target_weight_shortoptimistic_policy,
                global_weight,
            ).sum()

        assert len(loss_shortoptimistic_policy.shape) == 0
        assert len(target_weight_shortoptimistic_policy.shape) == 1
        assert target_weight_shortoptimistic_policy.shape[0] == n
        target_weight_shortoptimistic_policy_sum = (global_weight * target_weight_shortoptimistic_policy).sum()


        loss_value = self.loss_value_samplewise(
            value_logits, target_value, target_weight_value, global_weight
        ).sum()

        loss_td_value_unsummed = self.loss_td_value_samplewise(
            td_value_logits, target_td_value, target_weight_td_value, global_weight
        )
        assert self.num_td_values == 3
        loss_td_value1 = loss_td_value_unsummed[:,0].sum()
        loss_td_value2 = loss_td_value_unsummed[:,1].sum()
        loss_td_value3 = loss_td_value_unsummed[:,2].sum()

        loss_td_score = self.loss_td_score_samplewise(
            pred_td_score, target_td_score, target_weight_ownership, global_weight
        ).sum()

        loss_ownership = self.loss_ownership_samplewise(
            ownership_pretanh,
            target_ownership,
            target_weight_ownership_slot,
            mask,
            mask_sum_hw,
            global_weight,
        ).sum()
        loss_scoring = self.loss_scoring_samplewise(
            pred_scoring,
            target_scoring,
            target_weight_scoring,
            mask,
            mask_sum_hw,
            global_weight,
        ).sum()
        loss_futurepos = self.loss_futurepos_samplewise(
            futurepos_pretanh,
            target_futurepos,
            target_weight_futurepos,
            mask,
            mask_sum_hw,
            global_weight,
        ).sum()
        # Enclosure fork (owner, 2026-10-03): the seki slot holds edges that get cut (trainingwrite.cpp), with the
        # ownership target's weight (both come from the finished game), on rows that carry it (C68; earlier rows
        # have zeros there, which would teach that nothing is ever cut).
        loss_seki = self.loss_edgecut_samplewise(
            seki_logits,
            target_seki,
            target_weight_seki,
            mask,
            mask_sum_hw,
            global_weight,
        ).sum()
        seki_weight_scale = 1.0
        # Enclosure fork (owner, 2026-10-04): RGSC, on rows that carry a regret target (C71; C70 the regret).
        target_regret = target_global_nc[:, 70]
        target_weight_regret = target_global_nc[:, 71]
        loss_regret_value = self.loss_regret_value_samplewise(
            pred_regret, target_regret, target_weight_regret, global_weight,
        ).sum()
        loss_regret_ranking = self.loss_regret_ranking(pred_ranking, target_regret, target_weight_regret, global_weight)
        # Enclosure fork (2026-10-11): RGSC's swing regret, the same two losses on rows that carry it (C73; C72 the
        # swing regret, scaled to vanilla's range and capped at 4). A model without the swing heads outputs constants
        # there, so these losses train nothing in it.
        target_swing = target_global_nc[:, 72]
        target_weight_swing = target_global_nc[:, 73]
        loss_swing_value = self.loss_regret_value_samplewise(
            pred_swing, target_swing, target_weight_swing, global_weight,
        ).sum()
        loss_swing_ranking = self.loss_regret_ranking(pred_swing_ranking, target_swing, target_weight_swing, global_weight)
        loss_income = self.loss_income_samplewise(
            income_logits,
            target_income,
            target_weight_income,
            mask,
            mask_sum_hw,
            global_weight,
            None if self.income_mask is None else self.income_mask(input_binary_nchw),
        ).sum()
        loss_scoremean = self.loss_scoremean_samplewise(
            pred_scoremean,
            target_scoremean,
            target_weight_ownership,
            global_weight,
        ).sum()
        loss_scorebelief_cdf = self.loss_scorebelief_cdf_samplewise(
            scorebelief_logits,
            target_score_distribution,
            target_weight_ownership,
            global_weight,
        ).sum()
        loss_scorebelief_pdf = self.loss_scorebelief_pdf_samplewise(
            scorebelief_logits,
            target_score_distribution,
            target_weight_ownership,
            global_weight,
        ).sum()
        loss_scorestdev = self.loss_scorestdev_samplewise(
            pred_scorestdev,
            scorebelief_logits,
            global_weight,
        ).sum()
        loss_lead = self.loss_lead_samplewise(
            pred_lead,
            target_lead,
            target_weight_lead,
            global_weight,
        ).sum()
        loss_variance_time = self.loss_variance_time_samplewise(
            pred_variance_time,
            target_variance_time,
            target_weight_ownership,
            global_weight,
        ).sum()
        loss_shortterm_value_error = self.loss_shortterm_value_error_samplewise(
            pred_shortterm_value_error,
            td_value_logits,
            target_td_value,
            target_weight_ownership,
            global_weight,
        ).sum()
        loss_shortterm_score_error = self.loss_shortterm_score_error_samplewise(
            pred_shortterm_score_error,
            pred_td_score,
            target_td_score,
            target_weight_ownership,
            global_weight,
        ).sum()

        if not predict_q_values:
            target_weight_qvalues = torch.zeros_like(global_weight)
            loss_qvalues_winloss = torch.zeros_like(loss_policy_player)
            loss_qvalues_score = torch.zeros_like(loss_policy_player)
        else:
            target_qvalue_ncmove = batch["qValueTargetsNCMove"]
            target_weight_qvalues = torch.zeros_like(global_weight)
            loss_qvalues_winloss, loss_qvalues_score = self.loss_qvalues_samplewise(
                policy_logits[:, 6, :],
                policy_logits[:, 7, :],
                target_qvalue_ncmove[:, 0, :] / 32000.0,
                target_qvalue_ncmove[:, 1, :] / (60.0 / self.score_scale),  # Enclosure fork: stored as score * 60 / the score scale
                target_qvalue_ncmove[:, 2, :],
                global_weight
            )
            loss_qvalues_winloss = loss_qvalues_winloss.sum()
            loss_qvalues_score = loss_qvalues_score.sum()

        loss_sum = (
            loss_policy_player * policy_opt_loss_scale
            + loss_policy_opponent
            + loss_policy_player_soft * soft_policy_weight_scale
            + loss_policy_opponent_soft * soft_policy_weight_scale
            + loss_longoptimistic_policy * long_policy_opt_loss_scale
            + loss_shortoptimistic_policy * short_policy_opt_loss_scale
            + loss_value * value_loss_scale
            + loss_td_value1 * td_value_loss_scales[0]
            + loss_td_value2 * td_value_loss_scales[1]
            + loss_td_value3 * td_value_loss_scales[2]
            + loss_td_score
            + loss_ownership
            + loss_scoring * 0.25
            + loss_futurepos
            + loss_income
            + loss_regret_value
            + loss_regret_ranking
            + loss_swing_value
            + loss_swing_ranking
            + loss_seki * seki_loss_scale
            + loss_scoremean
            + loss_scorebelief_cdf
            + loss_scorebelief_pdf
            + loss_scorestdev
            + loss_lead
            + loss_variance_time * variance_time_loss_scale
            + loss_shortterm_value_error
            + loss_shortterm_score_error
            + loss_qvalues_winloss
            + loss_qvalues_score
        )

        policy_acc1 = self.accuracy1(
            policy_logits[:, 0, :],
            target_policy_player,
            target_weight_policy_player,
            global_weight,
        )
        square_value = self.square_value(value_logits, global_weight)

        results = {
            "p0loss_sum": loss_policy_player,
            "p1loss_sum": loss_policy_opponent,
            "p0softloss_sum": loss_policy_player_soft,
            "p1softloss_sum": loss_policy_opponent_soft,
            "p0lopt_sum": loss_longoptimistic_policy,
            "p0loptw_sum": target_weight_longoptimistic_policy_sum,
            "p0sopt_sum": loss_shortoptimistic_policy,
            "p0soptw_sum": target_weight_shortoptimistic_policy_sum,
            "vloss_sum": loss_value,
            "tdvloss1_sum": loss_td_value1,
            "tdvloss2_sum": loss_td_value2,
            "tdvloss3_sum": loss_td_value3,
            "tdsloss_sum": loss_td_score,
            "oloss_sum": loss_ownership,
            "sloss_sum": loss_scoring,
            "fploss_sum": loss_futurepos,
            "iloss_sum": loss_income,
            "rgloss_sum": loss_regret_value,
            "rkloss_sum": loss_regret_ranking,
            "rgrows_sum": (target_weight_regret * global_weight).sum(),
            "swloss_sum": loss_swing_value,
            "swkloss_sum": loss_swing_ranking,
            "swrows_sum": (target_weight_swing * global_weight).sum(),
            # Enclosure fork (2026-10-04): the share of rows carrying each Enclosure spatial target (logged as the
            # mean over rows, like the losses), so a loss divided by its share reads per row that carries the target
            # (these targets started on rows written from 2026-10-03, so the plain losses rose as such rows filled
            # the window; core/dashboard.py divides).
            "fpw_sum": (target_weight_futurepos * global_weight).sum(),
            "sw_sum": (target_weight_scoring * global_weight).sum(),
            "skw_sum": (target_weight_seki * global_weight).sum(),
            "iw_sum": (target_weight_income * global_weight).sum(),
            "skloss_sum": loss_seki,
            "smloss_sum": loss_scoremean,
            "sbcdfloss_sum": loss_scorebelief_cdf,
            "sbpdfloss_sum": loss_scorebelief_pdf,
            "sdregloss_sum": loss_scorestdev,
            "leadloss_sum": loss_lead,
            "vtimeloss_sum": loss_variance_time,
            "evstloss_sum": loss_shortterm_value_error,
            "esstloss_sum": loss_shortterm_score_error,
            "qwlloss_sum": loss_qvalues_winloss,
            "qscloss_sum": loss_qvalues_score,
            "loss_sum": loss_sum,
            "pacc1_sum": policy_acc1,
            "vsquare_sum": square_value,
        }

        if is_intermediate:
            return results
        else:
            weight = global_weight.sum()
            nsamples = int(global_weight.shape[0])
            policy_target_entropy = self.target_entropy(
                target_policy_player,
                target_weight_policy_player,
                global_weight,
            )
            soft_policy_target_entropy = self.target_entropy(
                target_policy_player_soft,
                target_weight_policy_player,
                global_weight,
            )

            extra_results = {
                "wsum": weight * self.world_size,
                "nsamp": nsamples * self.world_size,
                "ptentr_sum": policy_target_entropy,
                "ptsoftentr_sum": soft_policy_target_entropy,
                "sekiweightscale_sum": seki_weight_scale * weight,
            }

            if include_model_norms:
                extra_results.update(self.get_model_norm_metrics(raw_model))

            for key,value in extra_results.items():
                results[key] = value
            return results
