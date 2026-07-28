"""JEPA Implementation"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from .module import detach_clone


class PLDM(nn.Module):
    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

    def encode(self, info):
        """Encode observations and actions into embeddings.
        info: dict with pixels and action keys
        """
        pixels = info['pixels'].float()
        b = pixels.size(0)
        pixels = rearrange(
            pixels, 'b t ... -> (b t) ...'
        )  # flatten for encoding
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]  # cls token
        emb = self.projector(pixels_emb)
        info['emb'] = rearrange(emb, '(b t) d -> b t d', b=b)

        if 'action' in info:
            info['act_emb'] = self.action_encoder(info['action'])

        return info

    def predict(self, emb, act_emb):
        """Predict next state embedding
        emb: (B, T, D)
        act_emb: (B, T, A_emb)
        """
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, 'b t d -> (b t) d'))
        preds = rearrange(preds, '(b t) d -> b t d', b=emb.size(0))
        return preds

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = None):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the time horizon
        """
        if history_size is None:
            history_size = getattr(self.predictor, 'num_frames', 3)

        assert 'pixels' in info, 'pixels not in info_dict'
        H = info['pixels'].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info['action'] = act_0
        n_steps = T - H

        # copy and encode initial info dict
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v) and not k.endswith('_hist')}
        _init = self.encode(_init)
        emb = info['emb'] = _init['emb'].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        # flatten batch and sample dimensions for rollout
        emb = rearrange(emb, 'b s ... -> (b s) ...').clone()
        act = rearrange(act_0, 'b s ... -> (b s) ...')
        act_future = rearrange(act_future, 'b s ... -> (b s) ...')

        # rollout predictor autoregressively for n_steps
        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]  # (BS, HS, D)
            act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
            emb = torch.cat([emb, pred_emb], dim=1)  # (BS, T+1, D)

            next_act = act_future[:, t : t + 1, :]  # (BS, 1, action_dim)
            act = torch.cat([act, next_act], dim=1)  # (BS, T+1, action_dim)

        # predict the last state
        act_emb = self.action_encoder(act)  # (BS, T, A_emb)
        emb_trunc = emb[:, -HS:]  # (BS, HS, D)
        act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
        emb = torch.cat([emb, pred_emb], dim=1)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, '(b s) ... -> b s ...', b=B, s=S)
        info['predicted_emb'] = pred_rollout

        return info

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict['predicted_emb']  # (B,S, T-1, dim)
        goal_emb = info_dict['goal_emb']  # (B, T, dim)

        goal_emb = goal_emb[:, None, -1:, :].expand_as(pred_emb)

        # return last-step cost per action candidate
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction='none',
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """Compute the cost of action candidates given an info dict with goal and initial state."""

        assert 'goal' in info_dict, 'goal not in info_dict'

        # Same real-history injection as LeWM.get_cost -- see the long comment
        # there. PLDM.rollout uses the identical [H, T-H] split, so with H
        # frames and P planned blocks the tensor is T = H + P - 1 long: H-1 real
        # executed blocks then the plan. H=1 reproduces the old behavior.
        # read, never pop -- see the note in LeWM.get_cost
        px_hist = info_dict.get('pixels_hist')
        act_hist = info_dict.get('action_hist')
        if px_hist is not None and act_hist is not None:
            n_hist = px_hist.shape[2]
            if act_hist.shape[2] != n_hist - 1:
                raise ValueError(
                    f'action_hist has {act_hist.shape[2]} blocks but '
                    f'{n_hist} history frames need {n_hist - 1}'
                )
            info_dict['pixels'] = px_hist
            action_candidates = torch.cat(
                [act_hist.to(action_candidates), action_candidates], dim=2
            )
            # print once per process: proves the rollout really ran at H>1
            # rather than silently falling back to the single-frame path
            if not getattr(self, '_hist_inject_logged', False):
                self._hist_inject_logged = True
                print(
                    f'[hist-inject] H={n_hist} T={action_candidates.shape[2]} '
                    f'(expect T = H + horizon - 1)'
                )

        if 'goal_emb' not in info_dict:
            goal = {
                k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v) and not k.endswith('_hist')
            }
            goal['pixels'] = goal['goal']

            for k in info_dict:
                if k.startswith('goal_'):
                    goal[k[len('goal_') :]] = goal.pop(k)

            goal.pop('action')
            goal = self.encode(goal)

            info_dict['goal_emb'] = goal['emb']

        info_dict = self.rollout(info_dict, action_candidates)

        cost = self.criterion(info_dict)

        return cost


__all__ = ['PLDM']
