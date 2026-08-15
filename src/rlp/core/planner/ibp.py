"""IBPNet — the Imagination-Based Planner's manager, controller and memory.

Pascanu, Li, Vinyals, Heess, Buesing, Racaniere, Reichert, Weber, Wierstra,
Battaglia, *Learning model-based planning from scratch*, 2017
(`arXiv:1707.06170 <https://arxiv.org/abs/1707.06170>`_).

The paper's agent is three learned pieces around a model:

* a **manager** ``pi^M`` that emits, at every step, a discrete *route* from
  ``U = {act, s_j0, ..., s_jk}`` — either execute in the world, or imagine from
  a named node of the imagination tree.  Experiments restrict ``U`` to
  ``{act, s_j0, s_jk}`` (the real state, or the most recently imagined state),
  which is the route set implemented here.
* a **controller** ``pi^C`` that proposes an action from a state and the plan
  context.
* a **memory** ``mu`` (an LSTM) that aggregates every imagined step into the
  plan context conditioning both of the above.

The paper's fourth piece, the imagination ``I: S x A -> S x R``, is an
interaction network trained on-line from the agent's own experience.  Here it
is replaced by this repository's *frozen pretrained latent world model*, which
is what makes an IBP row comparable to the RLP/DMPO rows: identical data,
identical frozen critic, identical objective, differing only in the learned
planning procedure.  ``search`` therefore takes ``imagine`` and ``value`` as
callables and owns no dynamics of its own.

One node of the tree is a latent state; one *action* is one plan block (the
repository's ``action_block`` primitive steps), so a root-to-leaf chain of
``horizon`` blocks is exactly the plan the solver must return.  When the
manager routes to ``act`` before the chain is full, the controller completes it
greedily — see :meth:`IBPNet.search`.

Trainer: :mod:`rlp.train.ibp`.  Deployment: :class:`rlp.core.solver.IBPSolver`.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast

import torch
from torch import nn

__all__ = ["IBPNet", "SearchResult", "ROUTES"]

# The paper's tree-mode route set, restricted as in its experiments: act, or
# imagine from the real state (root) or from the last imagined state (chain).
ROUTES: tuple[str, ...] = ("act", "root", "last")

ACT, ROOT, LAST = 0, 1, 2


ImagineFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
ValueFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class SearchResult:
    """One decision's worth of planning."""

    plan: torch.Tensor  # (B, H, a_dim) the plan the manager committed to
    cost: torch.Tensor  # (B,) value of the executed plan's terminal node
    committed_cost: torch.Tensor  # (B,) value of the node `act` committed, before completion
    imagined: torch.Tensor  # (B,) imagination steps spent (the resource IBP pays for)
    unrolls: torch.Tensor  # (B,) world-model block unrolls, including greedy completion
    log_prob: torch.Tensor  # (B,) summed manager log-probability of the sampled routes
    entropy: torch.Tensor  # (B,) summed manager entropy (the paper's exploration bonus)
    routes: list[torch.Tensor] = field(default_factory=list)  # per-step route indices


def _mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 2):
        layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    return nn.Sequential(*layers)


class IBPNet(nn.Module):
    """Manager + controller + LSTM memory over a frozen imagination."""

    def __init__(
        self,
        z_dim: int,
        a_dim: int,
        horizon: int = 5,
        hidden: int = 256,
        memory: int = 256,
        amax: float = 2.5,
        max_imagine: int = 10,
        init_scale: float = 0.01,
    ) -> None:
        super().__init__()
        self.z_dim = int(z_dim)
        self.a_dim = int(a_dim)
        self.horizon = int(horizon)
        self.hidden = int(hidden)
        self.memory_dim = int(memory)
        self.amax = float(amax)
        self.max_imagine = int(max_imagine)

        # A node is encoded relative to the goal: the quasimetric critic reads
        # (state, goal), so the planner sees the same pair plus the difference.
        self.node_encoder = _mlp([3 * self.z_dim, hidden, hidden])
        # Memory input, the paper's per-step tuple (p, s_j, s_parent, a, s_child,
        # r, j, k) with states carried by their encodings and r the child's cost.
        step_dim = len(ROUTES) + 3 * hidden + self.a_dim + 3
        self.cell = nn.LSTMCell(step_dim, self.memory_dim)
        self.controller = _mlp([hidden + self.memory_dim, hidden, hidden, self.a_dim])
        # Manager sees both routable nodes, the plan context and the resource
        # state (depth, budget spent) it is asked to economize on.
        self.manager = _mlp([2 * hidden + self.memory_dim + 2, hidden, len(ROUTES)])
        for module in (self.controller, self.manager):
            last = module[-1]
            assert isinstance(last, nn.Linear)
            nn.init.normal_(last.weight, std=init_scale)
            nn.init.zeros_(last.bias)

    # ------------------------------------------------------------- pieces
    def encode_node(self, z: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.node_encoder(torch.cat([z, z_goal, z - z_goal], dim=-1)))

    def propose(self, node: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Controller action block, clipped to the plan box as ``amax * tanh``."""
        return self.amax * torch.tanh(self.controller(torch.cat([node, context], dim=-1)))

    def route_logits(
        self,
        root: torch.Tensor,
        last: torch.Tensor,
        context: torch.Tensor,
        depth: torch.Tensor,
        spent: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat([root, last, context, depth[:, None], spent[:, None]], dim=-1)
        return cast(torch.Tensor, self.manager(features))

    def initial_memory(self, batch: int, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
        zeros = torch.zeros(batch, self.memory_dim, device=device)
        return zeros, zeros.clone()

    # ------------------------------------------------------------- search
    def search(
        self,
        z_hist: torch.Tensor,
        a_hist: torch.Tensor,
        z_goal: torch.Tensor,
        imagine: ImagineFn,
        value: ValueFn,
        sample: bool = True,
        max_imagine: int | None = None,
    ) -> SearchResult:
        """Run one decision: imagine until the manager says ``act``.

        ``imagine(z_window, a_window, action) -> z_next`` advances the frozen
        world model by one plan block; ``value(z) -> cost`` scores a node
        against the goal (lower is better — the critic is a cost-to-go).

        The manager's ``act`` commits the best node found so far.  Because this
        repository's evaluation protocol executes a full ``horizon``-block plan
        open loop, a committed chain shorter than ``horizon`` is completed by
        running the controller greedily forward from that node; those unrolls
        are counted in :attr:`SearchResult.unrolls`.  Every batch element runs
        its own route sequence; elements that have acted are frozen by mask, so
        the batch costs the *slowest* element, not the sum.
        """
        budget = int(self.max_imagine if max_imagine is None else max_imagine)
        if budget < 1:
            raise ValueError("IBP needs at least one imagination step")
        batch, device = z_hist.shape[0], z_hist.device
        horizon, a_dim = self.horizon, self.a_dim

        h, c = self.initial_memory(batch, device)
        root_node = self.encode_node(z_hist[:, -1], z_goal)

        # chain = the most recently imagined branch; best = the committed one
        chain_z, chain_a = z_hist, a_hist
        chain_plan = torch.zeros(batch, horizon, a_dim, device=device)
        chain_len = torch.zeros(batch, dtype=torch.long, device=device)
        chain_node = root_node

        best_z, best_a = z_hist, a_hist
        best_plan = chain_plan
        best_len = chain_len
        best_cost = value(z_hist[:, -1])

        active = torch.ones(batch, dtype=torch.bool, device=device)
        imagined = torch.zeros(batch, device=device)
        unrolls = torch.zeros(batch, device=device)
        log_prob = torch.zeros(batch, device=device)
        entropy = torch.zeros(batch, device=device)
        routes: list[torch.Tensor] = []

        for step in range(budget):
            depth = chain_len.float() / horizon
            spent = torch.full_like(depth, step / budget)
            logits = self.route_logits(root_node, chain_node, h, depth, spent)
            # A full chain cannot be extended, and the first step has nothing to
            # commit, so those routes are masked out of the manager's choice.
            mask = torch.zeros_like(logits)
            mask[:, LAST] = torch.where(chain_len >= horizon, -torch.inf, 0.0)
            if step == 0:
                mask[:, ACT] = -torch.inf
            logits = logits + mask
            log_probs = torch.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            route = torch.multinomial(probs, 1).squeeze(-1) if sample else logits.argmax(dim=-1)
            chosen = log_probs.gather(1, route[:, None]).squeeze(-1)
            step_entropy = -(probs * log_probs.nan_to_num(neginf=0.0)).sum(dim=-1)
            log_prob = log_prob + torch.where(active, chosen, 0.0)
            entropy = entropy + torch.where(active, step_entropy, 0.0)
            routes.append(route)

            acting = (route == ACT) & active
            active = active & ~acting
            if not bool(active.any()):
                break

            from_root = (route == ROOT) | (chain_len >= horizon)
            select = from_root[:, None, None]
            parent_z = torch.where(select, z_hist, chain_z)
            parent_a = torch.where(select, a_hist, chain_a)
            parent_plan = torch.where(select, torch.zeros_like(chain_plan), chain_plan)
            parent_len = torch.where(from_root, torch.zeros_like(chain_len), chain_len)
            parent_node = torch.where(from_root[:, None], root_node, chain_node)

            action = self.propose(parent_node, h)
            child_z = imagine(parent_z, parent_a, action)
            child_cost = value(child_z)
            child_node = self.encode_node(child_z, z_goal)

            index = parent_len.clamp(max=horizon - 1)[:, None, None].expand(-1, 1, a_dim)
            child_plan = parent_plan.scatter(1, index, action[:, None])
            child_len = parent_len + 1

            keep = active[:, None, None]
            chain_z = torch.where(keep, torch.cat([parent_z, child_z[:, None]], dim=1)[:, -3:], chain_z)
            chain_a = torch.where(keep, torch.cat([parent_a, action[:, None]], dim=1)[:, -2:], chain_a)
            chain_plan = torch.where(keep, child_plan, chain_plan)
            chain_len = torch.where(active, child_len, chain_len)
            chain_node = torch.where(active[:, None], child_node, chain_node)

            better = active & (child_cost < best_cost)
            best_cost = torch.where(better, child_cost, best_cost)
            best_plan = torch.where(better[:, None, None], child_plan, best_plan)
            best_len = torch.where(better, child_len, best_len)
            best_z = torch.where(better[:, None, None], chain_z, best_z)
            best_a = torch.where(better[:, None, None], chain_a, best_a)
            imagined = imagined + active.float()
            unrolls = unrolls + active.float()

            one_hot = torch.nn.functional.one_hot(route, len(ROUTES)).float()
            counters = torch.stack([child_cost, depth, spent], dim=-1)
            features = torch.cat([one_hot, root_node, parent_node, action, child_node, counters], dim=-1)
            h_next, c_next = self.cell(features, (h, c))
            h = torch.where(active[:, None], h_next, h)
            c = torch.where(active[:, None], c_next, c)

        # Greedy completion: the protocol executes `horizon` blocks open loop.
        plan, node_z, node_a = best_plan, best_z, best_a
        node = self.encode_node(node_z[:, -1], z_goal)
        for _ in range(horizon):
            short = best_len < horizon
            if not bool(short.any()):
                break
            action = self.propose(node, h)
            child_z = imagine(node_z, node_a, action)
            index = best_len.clamp(max=horizon - 1)[:, None, None].expand(-1, 1, a_dim)
            filled = plan.scatter(1, index, action[:, None])
            plan = torch.where(short[:, None, None], filled, plan)
            node_z = torch.where(short[:, None, None], torch.cat([node_z, child_z[:, None]], 1)[:, -3:], node_z)
            node_a = torch.where(short[:, None, None], torch.cat([node_a, action[:, None]], 1)[:, -2:], node_a)
            node = self.encode_node(node_z[:, -1], z_goal)
            best_len = torch.where(short, best_len + 1, best_len)
            unrolls = unrolls + short.float()

        return SearchResult(
            plan=plan,
            cost=value(node_z[:, -1]),
            committed_cost=best_cost,
            imagined=imagined,
            unrolls=unrolls,
            log_prob=log_prob,
            entropy=entropy,
            routes=routes,
        )
