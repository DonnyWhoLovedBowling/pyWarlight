# warlight_policy_full.py
# Single-file: GraphSAGE encoder (depth=3), masked PlacementHead, Pre-LN AttackDecoder
# sequential sampling: placements -> apply -> re-encode -> attacks
# includes teacher-forced recompute helpers for PPO.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import List, Dict, Any, Tuple, Optional

from src.game.Phase import Phase
# ------------------------------
# GraphSAGE-style encoder (drop-in)
# ------------------------------
class GraphSAGEEncoder(nn.Module):
    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 hidden_dim: int = 128,
                 depth: int = 3,
                 skip_residuals: bool = False,
                 dropout: float = 0.1,
                 use_graphnorm: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.skip_residuals = skip_residuals
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.use_graphnorm = use_graphnorm

        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)

        self.msg_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ) for _ in range(depth)
        ])
        self.upd_mlps = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(2 * hidden_dim),
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ) for _ in range(depth)
        ])
        self.edge_upd_mlps = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(3 * hidden_dim),
                nn.Linear(3 * hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ) for _ in range(depth)
        ])

        if use_graphnorm:
            self.node_graphnorms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(depth)])
            self.edge_graphnorms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(depth)])
        else:
            self.node_graphnorms = nn.ModuleList([nn.Identity() for _ in range(depth)])
            self.edge_graphnorms = nn.ModuleList([nn.Identity() for _ in range(depth)])

        self.final_ln = nn.LayerNorm(hidden_dim)

    def forward(self, node_feats: torch.Tensor, edge_index: torch.Tensor, edge_feats: torch.Tensor):
        """
        node_feats: (N, F_node)
        edge_index: (2, E) long [src, dst]
        edge_feats: (E, F_edge)
        returns (node_h: (N,H), edge_h: (E,H))
        """
        device = node_feats.device
        src = edge_index[0].long().to(device)
        dst = edge_index[1].long().to(device)

        node_h = self.node_proj(node_feats)   # (N, H)
        edge_h = self.edge_proj(edge_feats)   # (E, H)

        E = edge_h.size(0)
        ones = torch.ones(E, device=device)

        for l in range(self.depth):
            # messages from src using edge features
            src_h = node_h[src]                # (E, H)
            msg_in = torch.cat([src_h, edge_h], dim=-1)  # (E, 2H)
            msg = self.msg_mlps[l](msg_in)     # (E, H)

            # aggregate (mean) at dst
            agg = torch.zeros_like(node_h)
            agg = agg.index_add(0, dst, msg)   # sum messages
            deg = torch.zeros(node_h.size(0), device=device).index_add(0, dst, ones)
            deg = deg.clamp(min=1.0).unsqueeze(-1)
            agg_mean = agg / deg

            # node update
            node_cat = torch.cat([node_h, agg_mean], dim=-1)
            node_upd = self.upd_mlps[l](node_cat)
            node_upd = self.dropout(node_upd)
            if not self.skip_residuals:
                node_h = node_h + node_upd
            else:
                node_h = node_upd
            node_h = self.node_graphnorms[l](node_h)

            # edge update (refine using updated node states)
            dst_h = node_h[dst]
            src_h = node_h[src]
            edge_cat = torch.cat([src_h, dst_h, edge_h], dim=-1)
            edge_upd = self.edge_upd_mlps[l](edge_cat)
            edge_upd = self.dropout(edge_upd)
            if not self.skip_residuals:
                edge_h = edge_h + edge_upd
            else:
                edge_h = edge_upd
            edge_h = self.edge_graphnorms[l](edge_h)

        node_h = self.final_ln(node_h)
        return node_h, edge_h

# ------------------------------
# PlacementHead: masked categorical draws (replacement sampling)
# ------------------------------
class PlacementHead(nn.Module):
    def __init__(self, node_hid_dim: int, edge_hid_dim: int, dropout: float = 0.1):
        super().__init__()
        # input is node_h (H) concatenated with global edge summary (H)
        self.fc = nn.Linear(node_hid_dim + edge_hid_dim, node_hid_dim)
        self.norm = nn.LayerNorm(node_hid_dim + edge_hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(node_hid_dim, 1)  # one logit per node after pooling

    def forward(self,
                node_embs: torch.Tensor,
                edge_embs: torch.Tensor,
                ownership_mask: torch.Tensor,
                n_available_armies: int) -> Tuple[torch.Tensor, List[int], torch.Tensor, torch.Tensor]:
        """
        node_embs: (N, H)
        edge_embs: (E, H) -- we will pool edge_embs to per-node summary
        ownership_mask: (N,) bool/int tensor (1 if owned)
        n_available_armies: int
        Returns:
            placement_counts: (N,) long tensor (counts per node)
            placements_list: list of length n_available_armies of sampled node indices
            masked_logits: (N,) tensor before softmax (masked -inf for illegal)
            probs: (N,) softmax probabilities
        """
        device = node_embs.device
        N = node_embs.size(0)

        # compute an edge summary and broadcast to nodes (simple mean)
        if edge_embs is None or edge_embs.numel() == 0:
            edge_summary = torch.zeros_like(node_embs)
        else:
            global_edge = edge_embs.mean(dim=0, keepdim=True)   # (1,H)
            edge_summary = global_edge.expand(N, -1)            # (N,H)

        x = torch.cat([node_embs, edge_summary], dim=-1)  # (N, H+H)
        x = self.norm(x)
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        logits_per_node = self.out(x).squeeze(-1)  # (N,)

        # mask out non-owned nodes
        ownership_mask_bool = ownership_mask.to(torch.bool)
        masked_logits = logits_per_node.masked_fill(~ownership_mask_bool, float('-inf'))

        probs = F.softmax(masked_logits, dim=-1)

        # sample n_available_armies draws with replacement from categorical(probs)
        if n_available_armies <= 0:
            placements = []
            placement_counts = torch.zeros(N, dtype=torch.long, device=device)
        else:
            placements_tensor = torch.multinomial(probs, n_available_armies, replacement=True)  # (n_available_armies,)
            placements = [int(x) for x in placements_tensor.tolist()]
            placement_counts = torch.bincount(placements_tensor, minlength=N).to(torch.long)

        return placement_counts, placements, masked_logits, probs

    def recompute_logps(self, node_embs: torch.Tensor, edge_embs: torch.Tensor,
                       ownership_mask: torch.Tensor, placements: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Teacher-forced recompute of per-draw logprobs and entropies (returns:
         logp_steps Tensor(shape num_draws), ent_steps Tensor(num_draws))
        """
        device = node_embs.device
        N = node_embs.size(0)

        if edge_embs is None or edge_embs.numel() == 0:
            edge_summary = torch.zeros_like(node_embs)
        else:
            global_edge = edge_embs.mean(dim=0, keepdim=True)
            edge_summary = global_edge.expand(N, -1)

        x = torch.cat([node_embs, edge_summary], dim=-1)
        x = self.norm(x)
        x = F.relu(self.fc(x))
        logits_per_node = self.out(x).squeeze(-1)
        ownership_mask_bool = ownership_mask.to(torch.bool)
        masked_logits = logits_per_node.masked_fill(~ownership_mask_bool, float('-inf'))
        
        # Check if all logits are -inf (no owned nodes), which would cause nan in softmax
        if torch.all(torch.isinf(masked_logits)):
            # No owned nodes - return empty tensors since no valid placements are possible
            if isinstance(placements, list) and len(placements) == 0:
                return torch.tensor([], device=device), torch.tensor([], device=device)
            elif isinstance(placements, torch.Tensor) and placements.numel() == 0:
                return torch.tensor([], device=device), torch.tensor([], device=device)
            else:
                # There are recorded placements but no owned nodes - this shouldn't happen
                # but we'll create a uniform distribution to avoid nan
                probs = torch.ones_like(masked_logits) / masked_logits.size(0)
        else:
            probs = F.softmax(masked_logits, dim=-1)
        
        dist = Categorical(probs=probs)

        # Handle different input formats for placements
        if isinstance(placements, list):
            if len(placements) == 0:
                return torch.tensor([], device=device), torch.tensor([], device=device)
            
            # Check if it's a list containing a batched tensor
            if len(placements) == 1 and isinstance(placements[0], torch.Tensor):
                # Extract the tensor and handle as batched data
                first_element = placements[0]
                if isinstance(first_element, torch.Tensor):
                    batched_tensor = first_element.to(device)  # [batch_size, max_placements]
                    if batched_tensor.dim() == 2:
                        # This is batched data - we can't process it directly in this method
                        # Return an error or handle appropriately
                        raise ValueError(f"Batched tensor input not supported in single-sample recompute_logps. "
                                       f"Tensor shape: {batched_tensor.shape}. Use batched processing instead.")
                    else:
                        # 1D tensor, treat as single sample
                        placements_tensor = batched_tensor
                else:
                    # This shouldn't happen due to the isinstance check, but for type safety
                    placements_tensor = torch.tensor(placements, device=device, dtype=torch.long)
            else:
                # Regular list of integers
                placements_tensor = torch.tensor(placements, device=device, dtype=torch.long)
        elif isinstance(placements, torch.Tensor):
            if placements.dim() == 0 and placements.numel() == 0:
                return torch.tensor([], device=device), torch.tensor([], device=device)
            placements_tensor = placements.to(device)
        else:
            raise ValueError(f"Unsupported placements type: {type(placements)}")
        
        # Filter out padding tokens (-1) if present
        if placements_tensor.numel() > 0:
            valid_mask = placements_tensor >= 0
            placements_tensor = placements_tensor[valid_mask]
        
        if placements_tensor.numel() == 0:
            return torch.tensor([], device=device), torch.tensor([], device=device)
        
        logp_steps = dist.log_prob(placements_tensor)
        ent_steps = dist.entropy().expand_as(logp_steps)
        return logp_steps, ent_steps

# ------------------------------
# AttackDecoder: Pre-LN Transformer + hierarchical frac bins -> deterministic mapping -> army_count
# ------------------------------
class PreLNBlock(nn.Module):
    def __init__(self, hidden_dim: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)

        self.ln3 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, 4*hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4*hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # Pre-LN self-attention
        h = self.ln1(tgt)
        sa_out, _ = self.self_attn(h, h, h, attn_mask=tgt_mask, need_weights=False)
        tgt = tgt + self.dropout1(sa_out)

        # Pre-LN cross-attention to memory
        h2 = self.ln2(tgt)
        ca_out, _ = self.cross_attn(h2, memory, memory, attn_mask=memory_mask, need_weights=False)
        tgt = tgt + self.dropout2(ca_out)

        # FFN (Pre-LN style)
        h3 = self.ln3(tgt)
        tgt = tgt + self.ff(h3)
        return tgt

class AttackDecoder(nn.Module):
    def __init__(self,
                 node_hid: int,
                 edge_hid: int,
                 num_edges: int,
                 frac_bins: int = 4,
                 max_steps: int = 20,
                 nhead: int = 4,
                 n_layers: int = 2,
                 dropout: float = 0.1):
        """
        - num_edges: number of directed edges E (edge_index length)
        - frac_bins: discrete fraction bins (e.g. 4 -> [0-25,25-50,50-75,75-100])
        """
        super().__init__()
        self.num_edges = num_edges
        self.frac_bins = frac_bins
        self.max_steps = max_steps
        self.input_fc = nn.Linear(node_hid + edge_hid, node_hid)
        self.tgt_ln = nn.LayerNorm(node_hid)
        self.mem_ln = nn.LayerNorm(node_hid)
        self.decoder_blocks = nn.ModuleList([PreLNBlock(node_hid, nhead, dropout) for _ in range(n_layers)])
        self.final_ln = nn.LayerNorm(node_hid)

        # heads
        self.edge_head = nn.Linear(node_hid, 1)      # score for each edge
        self.frac_head = nn.Linear(node_hid, frac_bins)      # fraction bin logits
        self.stop_head = nn.Linear(node_hid, 1)              # stop logit

    def forward(self,
            node_embs: torch.Tensor,
            edge_embs: torch.Tensor,
            edge_index: torch.Tensor,
            ownership_mask: torch.Tensor,
            armies_left: torch.Tensor) -> Tuple[Any, torch.Tensor, torch.Tensor]:
        """
        Supports both single-instance and batched inputs.

        Single mode:
            node_embs: (N, H)
            edge_embs: (E, H)
            ownership_mask: (N,)
            armies_left: (N,)

        Batch mode:
            node_embs: (B, N, H)
            edge_embs: (B, E, H)
            ownership_mask: (B, N)
            armies_left: (B, N)

        Returns for single mode:
            actions: dict with lists 'edge_idx','frac_idx','army_count'
            joint_logp: scalar tensor
            joint_ent: scalar tensor

        Returns for batch mode:
            actions: list of action-dicts (len B)
            joint_logp: tensor shape (B,)
            joint_ent: tensor shape (B,)
        """
        device = node_embs.device

        # Helper: run the original single-instance logic (adapted from your code)
        def _single_forward(node_embs_s, edge_embs_s, edge_index_s, ownership_mask_s, armies_left_s):
            """
            node_embs_s: (N,H)
            edge_embs_s: (E,H)
            ownership_mask_s: (N,)
            armies_left_s: (N,)
            """
            src = edge_index_s[0].long().to(device)
            dst = edge_index_s[1].long().to(device)
            E = edge_embs_s.size(0)
            N = node_embs_s.size(0)

            # memory: pooled node + edge contexts
            node_ctx = node_embs_s.mean(dim=0, keepdim=True)   # (1,H)
            edge_ctx = edge_embs_s.mean(dim=0, keepdim=True)   # (1,H)
            memory = torch.cat([node_ctx, edge_ctx], dim=-1)   # (1,2H)
            memory = self.input_fc(memory)                     # (1,H)
            memory = memory.unsqueeze(0)                       # (1,1,H)

            # per-edge projected context
            per_edge_inputs = torch.cat([node_embs_s[src], edge_embs_s], dim=-1)  # (E, 2H)
            per_edge_proj = self.input_fc(per_edge_inputs)  # (E, H)

            tgt = torch.zeros((1, 0, memory.size(-1)), device=device)  # (1, T, H)

            actions = {"edge_idx": [], "frac_idx": [], "army_count": []}
            logp_steps = []
            ent_steps = []

            # legal mask based on ownership and armies_left
            available_to_move = armies_left_s - 1
            available_to_move = torch.clamp(available_to_move, min=0)
            src_owned = ownership_mask_s[src].to(torch.bool)
            edge_legal_by_source = src_owned & (available_to_move[src] > 0)

            # fraction bins
            frac_bins = self.frac_bins
            frac_ranges = []
            step = 1.0 / frac_bins
            for i in range(frac_bins):
                low = i * step
                high = min(1.0, (i + 1) * step)
                frac_ranges.append((low, high))

            for _ in range(self.max_steps):
                # Decode pass
                out = tgt
                for block in self.decoder_blocks:
                    out = block(out if out.size(1) > 0 else memory, memory)

                if out.size(1) > 0:
                    last_out = out[:, -1, :]   # (1,H)
                else:
                    last_out = memory[:, 0, :]  # (1,H)

                last_out = self.final_ln(last_out)  # (1,H)

                # edge scores (E,)
                edge_scores = self.edge_head(per_edge_proj).squeeze(-1)  # (E,)
                masked_edge_scores = edge_scores.masked_fill(~edge_legal_by_source, float('-inf'))

                # stop score scalar
                stop_score = self.stop_head(last_out).squeeze(-1)  # scalar

                # unify shapes and concatenate -> (E+1,)
                masked_edge_scores = masked_edge_scores.view(-1)    # ensure 1D
                stop_score = stop_score.view(-1)                    # (1,)
                logits_with_stop = torch.cat([masked_edge_scores, stop_score], dim=0)  # (E+1,)
                # sample edge or stop
                edge_dist = Categorical(logits=logits_with_stop)
                edge_or_stop = int(edge_dist.sample().item())
                logp_edge = edge_dist.log_prob(torch.tensor(edge_or_stop, device=device))
                ent_edge = edge_dist.entropy()

                if edge_or_stop == E:
                    # STOP
                    logp_steps.append(logp_edge)
                    ent_steps.append(ent_edge)
                    break

                chosen_edge = edge_or_stop
                chosen_src = int(src[chosen_edge].item())
                available = int(available_to_move[chosen_src].item())

                # fraction selection
                frac_logits = self.frac_head(last_out).squeeze(0)  # (frac_bins,)
                frac_dist = Categorical(logits=frac_logits)
                frac_idx = int(frac_dist.sample().item())
                logp_frac = frac_dist.log_prob(torch.tensor(frac_idx, device=device))
                ent_frac = frac_dist.entropy()

                low, high = frac_ranges[frac_idx]
                frac_mid = (low + high) / 2.0
                army_choice = max(1, int(round(frac_mid * available))) if available > 0 else 0
                army_choice = min(army_choice, available) if available > 0 else 0

                # record
                actions["edge_idx"].append(chosen_edge)
                actions["frac_idx"].append(frac_idx)
                actions["army_count"].append(int(army_choice))

                # update armies_left and masks (non-in-place to preserve gradients)
                updated_armies = armies_left_s.clone()
                # Use scatter on a cloned tensor to avoid in-place writes on views
                index_tensor = torch.tensor([chosen_src], device=armies_left_s.device, dtype=torch.long)
                updated_value = torch.clamp(armies_left_s[chosen_src] - army_choice, min=0)
                updated_armies = updated_armies.scatter(0, index_tensor, updated_value.to(armies_left_s.dtype).unsqueeze(0))
                armies_left_s = updated_armies
                available_to_move = armies_left_s - 1
                available_to_move = torch.clamp(available_to_move, min=0)
                edge_legal_by_source = src_owned & (available_to_move[src] > 0)

                # logs
                logp_steps.append(logp_edge + logp_frac)
                ent_steps.append(ent_edge + ent_frac)

                # append autoregressive token
                last_token = last_out.unsqueeze(1)  # (1,1,H)
                tgt = torch.cat([tgt, last_token], dim=1)

            if len(logp_steps) == 0:
                joint_logp = torch.tensor(0.0, device=device)
                joint_ent = torch.tensor(0.0, device=device)
            else:
                joint_logp = torch.stack(logp_steps).sum()
                joint_ent = torch.stack(ent_steps).sum()

            return actions, joint_logp, joint_ent

        # -------------------------
        # Detect batch vs single
        # -------------------------
        if node_embs.dim() == 2:
            # single instance
            # make clones of armies_left because we update them in-place
            node_embs_s = node_embs
            edge_embs_s = edge_embs
            ownership_s = ownership_mask
            armies_left_s = armies_left.clone()
            return _single_forward(node_embs_s, edge_embs_s, edge_index, ownership_s, armies_left_s)

        elif node_embs.dim() == 3:
            # batched: loop over batch dimension and call single forward
            B = node_embs.size(0)
            actions_batch = []
            logp_list = []
            ent_list = []
            for b in range(B):
                node_b = node_embs[b]       # (N,H)
                edge_b = edge_embs[b]       # (E,H)
                own_b = ownership_mask[b]   # (N,)
                armies_b = armies_left[b].clone()  # (N,)
                acts, lp, ent = _single_forward(node_b, edge_b, edge_index, own_b, armies_b)
                actions_batch.append(acts)
                logp_list.append(lp)
                ent_list.append(ent)
            logp_tensor = torch.stack(logp_list) if len(logp_list) > 0 else torch.zeros((B,), device=device)
            ent_tensor = torch.stack(ent_list) if len(ent_list) > 0 else torch.zeros((B,), device=device)
            return actions_batch, logp_tensor, ent_tensor

        else:
            raise ValueError("node_embs.dim() must be 2 (single) or 3 (batch)")

    def recompute_logprobs(self,
                           node_embs: torch.Tensor,
                           edge_embs: torch.Tensor,
                           edge_index: torch.Tensor,
                           ownership_mask: torch.Tensor,
                           armies_left: torch.Tensor,
                           prev_actions: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Teacher-forcing forward pass for attack/transfer actions.

        Args:
            node_embs: (N,H) node embeddings
            edge_embs: (E,H) edge embeddings
            edge_index: (2,E) edge indices
            ownership_mask: (N,) bool/int for node ownership
            armies_left: (N,) int tensor (current armies at nodes)
            prev_actions: dictionary with keys:
                'edge_idx': list of chosen edges
                'frac_idx': list of fraction indices (optional)
                'army_count': list of army counts used

        Returns:
            joint_logp: scalar tensor of summed log-probabilities
            joint_entropy: scalar tensor of summed entropies
        """
        device = node_embs.device
        src = edge_index[0].long().to(device)
        N, E = node_embs.size(0), edge_embs.size(0)
        T = len(prev_actions['edge_idx'])

        # Memory
        node_ctx = node_embs.mean(dim=0, keepdim=True)
        edge_ctx = edge_embs.mean(dim=0, keepdim=True)
        memory = torch.cat([node_ctx, edge_ctx], dim=-1)
        memory = self.input_fc(memory).unsqueeze(0)  # (1,1,H)

        # per-edge context
        per_edge_inputs = torch.cat([node_embs[src], edge_embs], dim=-1)
        per_edge_proj = self.input_fc(per_edge_inputs)  # (E,H)

        # Edge logits once
        edge_scores = self.edge_head(per_edge_proj).squeeze(-1)  # (E,)
        src_owned = ownership_mask[src].to(torch.bool)
        available_to_move = torch.clamp(armies_left[src] - 1, min=0)
        edge_legal_mask = src_owned & (available_to_move > 0)
        masked_edge_scores = edge_scores.masked_fill(~edge_legal_mask, float('-inf'))

        # Stop token logits (broadcasted T times)
        stop_scores = self.stop_head(memory[:,0,:]).view(1)  # (1,)
        logits_with_stop = torch.cat([masked_edge_scores, stop_scores], dim=0)  # (E+1,)

        # Gather edge log-probs and entropy in one step
        edge_dist = Categorical(logits=logits_with_stop)
        chosen_edges = torch.tensor(prev_actions['edge_idx'], device=device)  # (T,)
        logp_edge = edge_dist.log_prob(chosen_edges)  # (T,)
        ent_edge = edge_dist.entropy()  # scalar, same for all steps
        ent_edge_steps = ent_edge.expand_as(logp_edge)  # (T,) - replicate for each step

        # Fraction logits
        frac_logits = self.frac_head(memory[:,0,:]).squeeze(0)  # (frac_bins,)
        if 'frac_idx' in prev_actions:
            chosen_fracs = torch.tensor(prev_actions['frac_idx'], device=device)  # (T,)
            frac_dist = Categorical(logits=frac_logits)
            logp_frac = frac_dist.log_prob(chosen_fracs)  # (T,)
            ent_frac = frac_dist.entropy()  # scalar
            ent_frac_steps = ent_frac.expand_as(logp_frac)  # (T,)
            logp_steps = logp_edge + logp_frac  # (T,)
            ent_steps = ent_edge_steps + ent_frac_steps  # (T,)
        else:
            logp_steps = logp_edge  # (T,)
            ent_steps = ent_edge_steps  # (T,)

        return logp_steps, ent_steps
# ------------------------------
# ValueHead (multi-layer with LayerNorm + Dropout)
# ------------------------------
class ValueHead(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, node_embs: torch.Tensor):
        g = node_embs.mean(dim=0, keepdim=True)
        g = self.norm1(g)
        g = F.relu(self.fc1(g))
        g = self.dropout(g)
        g = self.norm2(g)
        v = self.fc2(g)
        return v.squeeze()

# ------------------------------
# Full Policy wrapper
# ------------------------------
class WarlightPolicy(nn.Module):
    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 num_nodes: int,
                 num_edges: int,
                 hidden_dim: int = 128,
                 skip_residuals: bool = False):
        super().__init__()
        self.edge_index: Optional[torch.Tensor] = None
        self.encoder = GraphSAGEEncoder(node_dim, edge_dim, hidden_dim, depth=3, skip_residuals=skip_residuals)
        self.placement = PlacementHead(hidden_dim, hidden_dim)
        self.attack_decoder = AttackDecoder(hidden_dim, hidden_dim, num_edges)
        self.value_head = ValueHead(hidden_dim)

    # sequential sampling helper (recommended flow for rollout)
    @torch.no_grad()
    def forward(self,
                                       node_feats: torch.Tensor,
                                       edge_feats: torch.Tensor,
                                       ownership_mask: torch.Tensor,
                                       n_available_armies: int,
                                       armies_left: torch.Tensor,
                                       action: Phase) -> Dict[str, Any]:
        """
        1) encode state
        2) sample placements (only on owned regions)
        3) apply placements to node_feats and armies_left
        4) re-encode
        5) sample attacks autoregressively (masked by updated armies_left)
        Returns a dict with placements, placement_counts, attacks, joint logp, entropies, value, updated armies_left.
        """
        device = node_feats.device
        # encode original
        node_embs, edge_embs = self.encoder(node_feats, self.edge_index, edge_feats)
        if action == Phase.PLACE_ARMIES:
            # placements
            placement_counts, placements_list, placement_masked_logits, placement_probs = \
                self.placement(node_embs, edge_embs, ownership_mask, n_available_armies)
            # recompute placement logp/entropy for storage
            place_dist = Categorical(probs=placement_probs)
            if len(placements_list) > 0:
                placement_logp = place_dist.log_prob(torch.tensor(placements_list, device=device)).sum()
                placement_ent = place_dist.entropy().sum()
            else:
                placement_logp = torch.tensor(0.0, device=device)
                placement_ent = torch.tensor(0.0, device=device)
            out = {
                "placements_list": placements_list,
                "placement_counts": placement_counts,
                "logp": placement_logp,
                "ent": placement_ent,
            }

        elif action == Phase.ATTACK_TRANSFER:
            # attacks (attack decoder consumes ownership_mask and armies_left)
            attacks, attack_logp, attack_ent = self.attack_decoder(node_embs, edge_embs, self.edge_index,
                                                                ownership_mask, armies_left)
            # attacks is a dict with keys: edge_idx, frac_idx, army_count
            out = {
                "attacks": attacks,  # includes 'edge_idx', 'frac_idx', 'army_count'
                "logp": attack_logp,
                "ent": attack_ent,
            }
        else:
            raise ValueError(f"Unknown action phase: {action}")
        return out
    
    def get_value(self,
                  node_feats: torch.Tensor,
                  edge_feats: torch.Tensor,
                  ) -> torch.Tensor:
        """
        Get the value estimate for the current state.
        Supports both single-sample and batched inputs.
        
        Args:
            node_feats: [N, F] for single sample or [B, N, F] for batch
            edge_feats: [E, F_edge] for single sample or [B, E, F_edge] for batch
        
        Returns:
            value: [N] for single sample or [B, N] for batch
        """
        if self.edge_index is None:
            raise RuntimeError("edge_index must be initialized before calling get_value")
        
        # Handle both single sample and batched inputs
        if node_feats.dim() == 2:
            # Single sample case
            node_embs, edge_embs = self.encoder(node_feats, self.edge_index, edge_feats)
            value = self.value_head(node_embs)
            return value
        elif node_feats.dim() == 3:
            # Batched case - use mega-graph approach
            device = node_feats.device
            batch_size, num_nodes, node_feat_dim = node_feats.shape
            _, num_edges, edge_feat_dim = edge_feats.shape
            
            # Create mega-graph for batched encoding
            mega_node_feats = node_feats.view(batch_size * num_nodes, node_feat_dim)  # [B*N, F]
            mega_edge_feats = edge_feats.view(batch_size * num_edges, edge_feat_dim)  # [B*E, F_edge]
            
            # Create mega edge_index by offsetting node indices for each batch
            mega_edge_index = []
            for b in range(batch_size):
                offset_edge_index = self.edge_index + b * num_nodes
                mega_edge_index.append(offset_edge_index)
            mega_edge_index = torch.cat(mega_edge_index, dim=1)  # [2, B*E]
            
            # Vectorized encoder call
            mega_node_embs, mega_edge_embs = self.encoder(mega_node_feats, mega_edge_index, mega_edge_feats)
            hidden_dim = mega_node_embs.shape[1]
            
            # Reshape back to batched format and compute values
            node_embs_batch = mega_node_embs.view(batch_size, num_nodes, hidden_dim)  # [B, N, H]
            
            # Compute value for each sample in the batch
            values = []
            for i in range(batch_size):
                value = self.value_head(node_embs_batch[i])  # [N]
                values.append(value)
            
            return torch.stack(values)  # [B, N]
        else:
            raise ValueError(f"node_feats must be 2D or 3D tensor, got {node_feats.dim()}D")

    # teacher-forced recompute for PPO
    def recompute_turn_logprobs(self,
                               node_feats: torch.Tensor,
                               edge_feats: torch.Tensor,
                               ownership_mask: torch.Tensor,
                               placements_list: List[int],
                               attacks_recorded: Dict[str, Any],
                               armies_left_before: torch.Tensor,
                               action = None) -> Dict[str, Any]:
        """
        Recompute log-probs and entropies for the recorded turn.
        If action == Phase.PLACE_ARMIES: only placement logprobs/entropy.
        If action == Phase.ATTACK_TRANSFER: only attack logprobs/entropy.
        If action is None: do both (default).
        Returns per-step tensors and joint sums and final value.
        
        Handles both single-sample and batched inputs efficiently.
        """
        device = node_feats.device
        
        # Handle batched inputs efficiently
        if node_feats.dim() == 3:  # Batched input [batch_size, num_nodes, features]
            return self._recompute_batched_vectorized(
                node_feats, edge_feats, ownership_mask, 
                placements_list, attacks_recorded, armies_left_before, action
            )
        else:
            # Single sample input - process directly
            return self._recompute_single_sample(
                node_feats, edge_feats, ownership_mask, 
                placements_list, attacks_recorded, armies_left_before, action
            )
    
    def _recompute_batched_vectorized(self,
                                    node_feats: torch.Tensor,  # [B, N, F]
                                    edge_feats: torch.Tensor,  # [B, E, F_edge]
                                    ownership_mask: torch.Tensor,  # [B, N]
                                    placements_list: List,
                                    attacks_recorded,
                                    armies_left_before: torch.Tensor,  # [B, N]
                                    action = None) -> Dict[str, Any]:
        """
        Fully vectorized processing of batched inputs using mega-graph approach.
        Takes advantage of the fact that edge_index is constant across all samples.
        """
        device = node_feats.device
        batch_size, num_nodes, node_feat_dim = node_feats.shape
        _, num_edges, edge_feat_dim = edge_feats.shape
        
        # Validate input dimensions for batched processing
        if ownership_mask.dim() == 1:
            # ownership_mask is not batched - expand it for all samples
            if ownership_mask.size(0) != num_nodes:
                raise ValueError(f"ownership_mask size {ownership_mask.size(0)} doesn't match num_nodes {num_nodes}")
            ownership_mask = ownership_mask.unsqueeze(0).expand(batch_size, -1)  # [1, N] -> [B, N]
        elif ownership_mask.dim() == 2:
            if ownership_mask.shape != (batch_size, num_nodes):
                raise ValueError(f"ownership_mask shape {ownership_mask.shape} doesn't match expected ({batch_size}, {num_nodes})")
        else:
            raise ValueError(f"ownership_mask must have 1 or 2 dimensions, got {ownership_mask.dim()}")
        
        # Similar validation for armies_left_before
        if armies_left_before.dim() == 1:
            if armies_left_before.size(0) != num_nodes:
                raise ValueError(f"armies_left_before size {armies_left_before.size(0)} doesn't match num_nodes {num_nodes}")
            armies_left_before = armies_left_before.unsqueeze(0).expand(batch_size, -1)  # [1, N] -> [B, N]
        elif armies_left_before.dim() == 2:
            if armies_left_before.shape != (batch_size, num_nodes):
                raise ValueError(f"armies_left_before shape {armies_left_before.shape} doesn't match expected ({batch_size}, {num_nodes})")
        else:
            raise ValueError(f"armies_left_before must have 1 or 2 dimensions, got {armies_left_before.dim()}")
        
        # Ensure edge_index is set
        if self.edge_index is None:
            raise RuntimeError("edge_index must be initialized before calling recompute_turn_logprobs")
        
        # Determine action mode
        if action is None:
            action_mode = "both"
        elif action == Phase.PLACE_ARMIES:
            action_mode = "placement"
        elif action == Phase.ATTACK_TRANSFER:
            action_mode = "attack"
        else:
            action_mode = "both"
        
        # Create mega-graph for initial encoding
        mega_node_feats = node_feats.view(batch_size * num_nodes, node_feat_dim)  # [B*N, F]
        mega_edge_feats = edge_feats.view(batch_size * num_edges, edge_feat_dim)  # [B*E, F_edge]
        
        # Create mega edge_index by offsetting node indices for each batch
        mega_edge_index = []
        for b in range(batch_size):
            offset_edge_index = self.edge_index + b * num_nodes
            mega_edge_index.append(offset_edge_index)
        mega_edge_index = torch.cat(mega_edge_index, dim=1)  # [2, B*E]
        
        # First vectorized encoder call
        mega_node_embs, mega_edge_embs = self.encoder(mega_node_feats, mega_edge_index, mega_edge_feats)
        hidden_dim = mega_node_embs.shape[1]
        
        # Reshape back to batched format
        node_embs_batch = mega_node_embs.view(batch_size, num_nodes, hidden_dim)  # [B, N, H]
        edge_embs_batch = mega_edge_embs.view(batch_size, num_edges, hidden_dim)  # [B, E, H]
        
        # Process placements for all samples
        placement_results = []
        for i in range(batch_size):
            # Handle different input formats for placements_list
            if isinstance(placements_list, list) and len(placements_list) == 1 and isinstance(placements_list[0], torch.Tensor):
                # placements_list is [tensor([batch_size, max_placements])]
                batched_placements = placements_list[0].to(device)  # [batch_size, max_placements]
                if batched_placements.dim() == 2:
                    # Extract i-th sample and remove padding
                    sample_placements = batched_placements[i]  # [max_placements]
                    valid_mask = sample_placements >= 0
                    single_placements = sample_placements[valid_mask].tolist()
                else:
                    # 1D tensor case
                    single_placements = batched_placements.tolist()
            elif isinstance(placements_list, list) and len(placements_list) > 0 and isinstance(placements_list[0], list):
                # placements_list is [[sample0], [sample1], ...]
                single_placements = placements_list[i] if i < len(placements_list) else []
            elif isinstance(placements_list, list):
                # placements_list is a single list to be used for all samples
                single_placements = placements_list
            else:
                # Fallback
                single_placements = []
            
            # Ensure single_placements is a list
            if not isinstance(single_placements, list):
                single_placements = [single_placements] if single_placements is not None else []
            
            place_logp_steps, place_ent_steps = self.placement.recompute_logps(
                node_embs_batch[i], edge_embs_batch[i], ownership_mask[i], single_placements
            )
            placement_results.append({
                "place_logp_steps": place_logp_steps,
                "place_ent_steps": place_ent_steps,
                "placements": single_placements
            })
        
        # If placement-only mode, return early
        if action_mode == "placement":
            results = []
            for i, place_result in enumerate(placement_results):
                placement_logp_new = place_result["place_logp_steps"].sum() if place_result["place_logp_steps"].numel() > 0 else torch.tensor(0.0, device=device)
                placement_ent_new = place_result["place_ent_steps"].sum() if place_result["place_ent_steps"].numel() > 0 else torch.tensor(0.0, device=device)
                value = self.value_head(node_embs_batch[i])
                results.append({
                    "place_logp_steps": place_result["place_logp_steps"],
                    "place_ent_steps": place_result["place_ent_steps"],
                    "placement_logp_new": placement_logp_new,
                    "joint_logp_new": placement_logp_new,
                    "joint_ent_new": placement_ent_new,
                    "value": value
                })
            return self._aggregate_batch_results(results, device)
        
        # For attack or both modes, we need updated features after placements
        results = []  # Initialize results list
        if action_mode in ["attack", "both"]:
            # Apply placements to all samples and create mega-graph for re-encoding
            updated_node_feats_batch = []
            updated_armies_left_batch = []
            
            for i, place_result in enumerate(placement_results):
                single_placements = place_result["placements"]
                placement_counts = torch.bincount(
                    torch.tensor(single_placements, device=device, dtype=torch.long), 
                    minlength=num_nodes
                ).to(node_feats.dtype)
                
                updated_node_feats = node_feats[i].clone()
                # Non-in-place update to preserve gradients
                army_column = updated_node_feats[:, 0].clone()
                updated_army_feats = army_column + placement_counts
                zero_column_index = torch.zeros((updated_node_feats.shape[0], 1), dtype=torch.long, device=device)
                updated_node_feats = updated_node_feats.scatter(1, zero_column_index, updated_army_feats.unsqueeze(1))
                updated_armies_left = armies_left_before[i].clone().to(device) + placement_counts.to(armies_left_before.dtype)
                
                updated_node_feats_batch.append(updated_node_feats)
                updated_armies_left_batch.append(updated_armies_left)
            
            # Create mega-graph for re-encoding with updated features
            updated_node_feats_stacked = torch.stack(updated_node_feats_batch)  # [B, N, F]
            mega_updated_node_feats = updated_node_feats_stacked.view(batch_size * num_nodes, node_feat_dim)
            
            # Vectorized re-encoding (this is the key optimization!)
            mega_node_embs2, mega_edge_embs2 = self.encoder(mega_updated_node_feats, mega_edge_index, mega_edge_feats)
            
            # Reshape back
            node_embs2_batch = mega_node_embs2.view(batch_size, num_nodes, hidden_dim)
            edge_embs2_batch = mega_edge_embs2.view(batch_size, num_edges, hidden_dim)
            
            # Process attacks for all samples (still need per-sample due to complex logic)
            results = []
            for i in range(batch_size):
                single_attacks = attacks_recorded if isinstance(attacks_recorded, dict) else attacks_recorded[i] if isinstance(attacks_recorded, list) else {}
                
                attack_logp_steps, attack_ent_steps = self.attack_decoder.recompute_logprobs(
                    node_embs2_batch[i], edge_embs2_batch[i], self.edge_index, 
                    ownership_mask[i], updated_armies_left_batch[i], single_attacks
                )
                
                attack_logp_new = attack_logp_steps.sum() if attack_logp_steps.numel() > 0 else torch.tensor(0.0, device=device)
                attack_ent_new = attack_ent_steps.sum() if attack_ent_steps.numel() > 0 else torch.tensor(0.0, device=device)
                
                if action_mode == "attack":
                    # Attack-only mode
                    value = self.value_head(node_embs2_batch[i])
                    results.append({
                        "attack_logp_steps": attack_logp_steps,
                        "attack_ent_steps": attack_ent_steps,
                        "attack_logp_new": attack_logp_new,
                        "joint_logp_new": attack_logp_new,
                        "joint_ent_new": attack_ent_new,
                        "value": value
                    })
                else:
                    # Both mode
                    place_result = placement_results[i]
                    placement_logp_new = place_result["place_logp_steps"].sum() if place_result["place_logp_steps"].numel() > 0 else torch.tensor(0.0, device=device)
                    placement_ent_new = place_result["place_ent_steps"].sum() if place_result["place_ent_steps"].numel() > 0 else torch.tensor(0.0, device=device)
                    value = self.value_head(node_embs2_batch[i])
                    
                    results.append({
                        "place_logp_steps": place_result["place_logp_steps"],
                        "place_ent_steps": place_result["place_ent_steps"],
                        "attack_logp_steps": attack_logp_steps,
                        "attack_ent_steps": attack_ent_steps,
                        "placement_logp_new": placement_logp_new,
                        "attack_logp_new": attack_logp_new,
                        "joint_logp_new": placement_logp_new + attack_logp_new,
                        "joint_ent_new": placement_ent_new + attack_ent_new,
                        "value": value
                    })
        
        return self._aggregate_batch_results(results, device)
    
    def _recompute_single_sample(self,
                                node_feats: torch.Tensor,
                                edge_feats: torch.Tensor,
                                ownership_mask: torch.Tensor,
                                placements_list: List[int],
                                attacks_recorded: Dict[str, Any],
                                armies_left_before: torch.Tensor,
                                action = None) -> Dict[str, Any]:
        """Process a single sample (non-batched)"""
        device = node_feats.device
        
        # Ensure edge_index is set
        if self.edge_index is None:
            raise RuntimeError("edge_index must be initialized before calling recompute_turn_logprobs")
            
        node_embs, edge_embs = self.encoder(node_feats, self.edge_index, edge_feats)

        # Default: do both
        if action is None:
            action_mode = "both"
        elif action == Phase.PLACE_ARMIES:
            action_mode = "placement"
        elif action == Phase.ATTACK_TRANSFER:
            action_mode = "attack"
        else:
            action_mode = "both"

        # Placement only
        if action_mode == "placement":
            place_logp_steps, place_ent_steps = self.placement.recompute_logps(node_embs, edge_embs,
                                                                               ownership_mask, placements_list)
            placement_logp_new = place_logp_steps.sum() if place_logp_steps.numel() > 0 else torch.tensor(0.0, device=device)
            placement_ent_new = place_ent_steps.sum() if place_ent_steps.numel() > 0 else torch.tensor(0.0, device=device)
            value = self.value_head(node_embs)
            return {
                "place_logp_steps": place_logp_steps,
                "place_ent_steps": place_ent_steps,
                "placement_logp_new": placement_logp_new,
                "joint_logp_new": placement_logp_new,
                "joint_ent_new": placement_ent_new,
                "value": value
            }

        # Attack only
        if action_mode == "attack":
            # Apply placements deterministically
            placement_counts = torch.bincount(torch.tensor(placements_list, device=device, dtype=torch.long), minlength=node_feats.size(0)).to(node_feats.dtype)
            updated_node_feats = node_feats.clone()
            # Non-in-place update to preserve gradients  
            army_column = updated_node_feats[:, 0].clone()
            updated_army_feats = army_column + placement_counts
            zero_column_index = torch.zeros((updated_node_feats.shape[0], 1), dtype=torch.long, device=device)
            updated_node_feats = updated_node_feats.scatter(1, zero_column_index, updated_army_feats.unsqueeze(1))
            updated_armies_left = armies_left_before.clone().to(device) + placement_counts.to(armies_left_before.dtype)
            node_embs2, edge_embs2 = self.encoder(updated_node_feats, self.edge_index, edge_feats)
            attack_logp_steps, attack_ent_steps = self.attack_decoder.recompute_logprobs(node_embs2, edge_embs2,
                                                                                         self.edge_index, ownership_mask,
                                                                                         updated_armies_left, attacks_recorded)
            attack_logp_new = attack_logp_steps.sum() if attack_logp_steps.numel() > 0 else torch.tensor(0.0, device=device)
            attack_ent_new = attack_ent_steps.sum() if attack_ent_steps.numel() > 0 else torch.tensor(0.0, device=device)
            value = self.value_head(node_embs2)
            return {
                "attack_logp_steps": attack_logp_steps,
                "attack_ent_steps": attack_ent_steps,
                "attack_logp_new": attack_logp_new,
                "joint_logp_new": attack_logp_new,
                "joint_ent_new": attack_ent_new,
                "value": value
            }

        # Both (default)
        # placements recompute
        place_logp_steps, place_ent_steps = self.placement.recompute_logps(node_embs, edge_embs,
                                                                           ownership_mask, placements_list)
        placement_logp_new = place_logp_steps.sum() if place_logp_steps.numel() > 0 else torch.tensor(0.0, device=device)
        placement_ent_new = place_ent_steps.sum() if place_ent_steps.numel() > 0 else torch.tensor(0.0, device=device)

        # apply placements deterministically
        placement_counts = torch.bincount(torch.tensor(placements_list, device=device, dtype=torch.long), minlength=node_feats.size(0)).to(node_feats.dtype)
        updated_node_feats = node_feats.clone()
        # Non-in-place update to preserve gradients
        army_column = updated_node_feats[:, 0].clone()
        updated_army_feats = army_column + placement_counts
        zero_column_index = torch.zeros((updated_node_feats.shape[0], 1), dtype=torch.long, device=device)
        updated_node_feats = updated_node_feats.scatter(1, zero_column_index, updated_army_feats.unsqueeze(1))

        updated_armies_left = armies_left_before.clone().to(device) + placement_counts.to(armies_left_before.dtype)

        # re-encode
        node_embs2, edge_embs2 = self.encoder(updated_node_feats, self.edge_index, edge_feats)

        # attacks recompute (teacher forced) — returns logp_steps, ent_steps
        attack_logp_steps, attack_ent_steps = self.attack_decoder.recompute_logprobs(node_embs2, edge_embs2,
                                                                                     self.edge_index, ownership_mask,
                                                                                     updated_armies_left, attacks_recorded)
        attack_logp_new = attack_logp_steps.sum() if attack_logp_steps.numel() > 0 else torch.tensor(0.0, device=device)
        attack_ent_new = attack_ent_steps.sum() if attack_ent_steps.numel() > 0 else torch.tensor(0.0, device=device)

        value = self.value_head(node_embs2)

        return {
            "place_logp_steps": place_logp_steps,
            "place_ent_steps": place_ent_steps,
            "attack_logp_steps": attack_logp_steps,
            "attack_ent_steps": attack_ent_steps,
            "placement_logp_new": placement_logp_new,
            "attack_logp_new": attack_logp_new,
            "joint_logp_new": placement_logp_new + attack_logp_new,
            "joint_ent_new": placement_ent_new + attack_ent_new,
            "value": value
        }

    def _aggregate_batch_results(self, results, device):
        """
        Aggregate results from batch processing into the expected format
        """
        # Extract and stack/concatenate results from all samples in batch
        batch_size = len(results)
        
        # Collect logprobs for each sample - these need to be returned as tensors with shape [batch_size, max_actions]  
        placement_logps = []
        attack_logps = []
        
        for result in results:
            if "place_logp_steps" in result and result["place_logp_steps"].numel() > 0:
                placement_logps.append(result["place_logp_steps"])
            else:
                placement_logps.append(torch.tensor([], device=device))
            
            if "attack_logp_steps" in result and result["attack_logp_steps"].numel() > 0:
                attack_logps.append(result["attack_logp_steps"])  
            else:
                attack_logps.append(torch.tensor([], device=device))
        
        # Pad and stack the log probabilities
        from .RLUtils import pad_tensor_list
        placement_logps_padded = pad_tensor_list(placement_logps, pad_value=0, target_device=device) if placement_logps else torch.empty((batch_size, 0), device=device)
        attack_logps_padded = pad_tensor_list(attack_logps, pad_value=0, target_device=device) if attack_logps else torch.empty((batch_size, 0), device=device)
        
        # Also collect entropy information
        placement_entropies = []
        attack_entropies = []
        joint_entropies = []
        
        for result in results:
            if "place_ent_steps" in result and result["place_ent_steps"].numel() > 0:
                placement_entropies.append(result["place_ent_steps"].sum())
            else:
                placement_entropies.append(torch.tensor(0.0, device=device))
                
            if "attack_ent_steps" in result and result["attack_ent_steps"].numel() > 0:
                attack_entropies.append(result["attack_ent_steps"].sum())
            else:
                attack_entropies.append(torch.tensor(0.0, device=device))
                
            if "joint_ent_new" in result:
                joint_entropies.append(result["joint_ent_new"])
            else:
                joint_entropies.append(torch.tensor(0.0, device=device))
        
        return {
            "logp": placement_logps_padded,  # For placement phase
            "attack_logp": attack_logps_padded,  # For attack phase
            "placement_entropy": torch.stack(placement_entropies).mean().item() if placement_entropies else 0.0,
            "attack_entropy": torch.stack(attack_entropies).mean().item() if attack_entropies else 0.0,
            "joint_entropy": torch.stack(joint_entropies).mean().item() if joint_entropies else 0.0
        }

    def ppo_loss(self,
                 joint_logp_new: torch.Tensor,
                 joint_logp_old: torch.Tensor,
                 advantages: torch.Tensor,
                 value_pred: torch.Tensor,
                 returns: torch.Tensor,
                 clip_eps: float = 0.2,
                 vf_coef: float = 0.5,
                 ent_coef: float = 0.01,
                 entropy = None) -> Dict[str, torch.Tensor]:
        ratio = torch.exp(joint_logp_new - joint_logp_old)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(value_pred, returns)
        ent_bonus = -ent_coef * (entropy.mean() if entropy is not None else torch.tensor(0.0, device=value_pred.device))
        loss = policy_loss + vf_coef * value_loss + ent_bonus
        return {
            "loss": loss,
            "policy_loss": policy_loss.detach(),
            "value_loss": value_loss.detach(),
            "entropy_bonus": (-ent_bonus).detach(),
            "ratio": ratio.detach().mean()
        }

# ------------------------------
# Minimal example
# ------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    # toy graph
    N = 8
    E = 16
    node_feat_dim = 6   # assume node_feats[:,0] = current armies
    edge_feat_dim = 4

    node_feats = torch.randn(N, node_feat_dim)
    edge_feats = torch.randn(E, edge_feat_dim)
    # sample edge_index (directed)
    edge_index = torch.randint(0, N, (2, E))

    # ownership mask: which nodes are owned by bot
    ownership_mask = torch.zeros(N, dtype=torch.int)
    ownership_mask[[0,2,4]] = 1

    # armies left per node (int tensor)
    armies_left = torch.randint(1, 6, (N,))  # between 1 and 5 armies

    n_available_armies = 5

    policy = WarlightPolicy(node_dim=node_feat_dim,
                            edge_dim=edge_feat_dim,
                            num_nodes=N,
                            num_edges=E,
                            hidden_dim=64,
                            skip_residuals=False)

    result = policy(node_feats, edge_feats,
                    ownership_mask, n_available_armies, 
                    armies_left, action=Phase.PLACE_ARMIES)

    print("Placements (per-draw list):", result["placements_list"])
    print("Placement counts per node:", result["placement_counts"].tolist())
    result = policy(node_feats, edge_feats,
                ownership_mask, n_available_armies, 
                armies_left, action=Phase.ATTACK_TRANSFER)

    print("Attacks:", result["attacks"])
    print("Attack joint logp:", result["logp"].item())
    # print("Placement joint logp:", result["placement_logp"].item())
    # print("Joint logp:", result["joint_logp"].item())
    # print("Entropy:", result["joint_ent"].item())
    # print("Value:", result["value"].item())
