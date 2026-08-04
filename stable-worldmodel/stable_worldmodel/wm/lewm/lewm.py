import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class LeWM(nn.Module):
    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        **kwargs,
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
        pixels = info['pixels'].to(next(self.encoder.parameters()).dtype)
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

        # encode initial state, or reuse cached embedding from a prior rollout.
        # detach: to avoid backprop in encoder
        if 'emb' not in info:
            _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v) and not k.endswith('_hist')}
            _init = self.encode(_init)
            info['emb'] = (
                _init['emb'].detach().unsqueeze(1).expand(B, S, -1, -1)
            )

        # flatten batch and sample dimensions for rollout
        emb_init = rearrange(info['emb'], 'b s ... -> (b s) ...')
        act_flat = rearrange(act_0, 'b s ... -> (b s) ...')
        act_future_flat = rearrange(act_future, 'b s ... -> (b s) ...')
        all_act_emb = self.action_encoder(
            torch.cat([act_flat, act_future_flat], dim=1)
        )  # (BS, T, A_emb)

        # rollout predictor autoregressively for n_steps + 1 (final) steps
        # emb_list holds individual (BS, D) frames, each with its own grad_fn
        HS = history_size
        emb_list = list(emb_init.unbind(dim=1))  # H tensors of shape (BS, D)
        for t in range(n_steps + 1):
            lo = max(0, H + t - HS)
            emb_trunc = torch.stack(emb_list[lo:], dim=1)  # (BS, HS, D)
            act_trunc = all_act_emb[:, lo : H + t]  # (BS, HS, A_emb)
            emb_list.append(self.predict(emb_trunc, act_trunc)[:, -1])

        emb = torch.stack(emb_list, dim=1)  # (BS, H + n_steps + 1, D)

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

        # Real conditioning history, so EVERY planner (CEM / MPPI / Adam) sees
        # the same 3 frames LIP does. Without this the sample-based planners get
        # H = info['pixels'].size(2) = 1 -- EnvPool hands over a single frame --
        # and grow the attention window with their OWN predictions, so they have
        # no velocity information either. One frame cannot show which way a
        # 2-joint arm is moving.
        #
        # `rollout` splits the action tensor as [H, T-H] and pairs action i with
        # state i, so with H frames and P planned blocks the tensor must be
        # T = H + P - 1 long: the first H-1 entries are the actions actually
        # executed into the observed states, the last P are the plan. At H=1
        # that is T = P, i.e. exactly the current behavior.
        # Read, do NOT pop: get_cost is called once per solver iteration (~30
        # for CEM) with the SAME dict, so popping made only the first call
        # inject. The rest then rolled a 3-frame `pixels` against a 5-block
        # action tensor -- a silently different rollout length, which cored.
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

        # encode goal state, or reuse cached embedding from a prior call
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


__all__ = ['LeWM']
