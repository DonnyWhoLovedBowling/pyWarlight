"""
Node Features Usage Summary - Verification

This document verifies that the three node feature sets are used correctly throughout the codebase.

=== THREE NODE FEATURE SETS ===

1. STARTING_NODE_FEATURES
   - Represents: Game state at the beginning of the turn (before placement)
   - When captured: In init_turn() before any actions
   - Used for: Placement decisions (where to place armies)

2. POST_PLACEMENT_NODE_FEATURES  
   - Represents: Game state after placement, before attacks/transfers
   - When captured: In attack_transfer() after armies are placed
   - Used for: Attack/transfer decisions (where to attack/transfer)

3. END_FEATURES
   - Represents: Game state after all actions are complete (end of turn)
   - When captured: In end_move() after all attacks/transfers are processed
   - Used for: Value computation (expected return from final state)

=== CURRENT USAGE VERIFICATION ===

✅ CORRECT USAGE:

RLGNNAgent.py:
- Line ~402: self.starting_node_features = torch.tensor(game.create_node_features()) in init_turn()
- Line ~464: self.post_placement_node_features = torch.tensor(game.create_node_features()) in attack_transfer()  
- Line ~640: end_features = torch.tensor(game.create_node_features()) in end_move()
- Line ~641: value = self.model.get_value(end_features) - CORRECT: Using end_features for value
- Line ~689: next_value = self.model.get_value(end_features) - CORRECT: Using end_features for GAE

PPOAgent.py:
- Line ~200: placement_logits = agent.run_model(starting_features_batched, PLACE_ARMIES) - CORRECT
- Line ~205: attack_logits = agent.run_model(post_features_batched, ATTACK_TRANSFER) - CORRECT  
- Line ~410: values_pred = self.policy.get_value(buffer.get_end_features()) - CORRECT

RolloutBuffer:
- Stores all three feature sets separately
- get_starting_node_features() returns starting features - CORRECT
- get_post_placement_node_features() returns post placement features - CORRECT
- get_end_features() returns end features - CORRECT

=== WHY THIS IS CORRECT ===

1. Placement Policy:
   - Should decide where to place armies based on the initial state
   - Uses starting_node_features ✅

2. Attack/Transfer Policy:
   - Should decide where to attack/transfer based on state after placement
   - Uses post_placement_node_features ✅

3. Value Function:
   - Should estimate expected return from the final state the agent reaches
   - Uses end_features ✅
   - This ensures consistency between rollout and training

4. GAE Computation:
   - Needs the value of the actual final state for advantage estimation
   - Uses end_features ✅

=== BENEFIT OF THIS APPROACH ===

- Placement policy sees the clean state before any turn actions
- Attack policy sees the state with newly placed armies  
- Value function evaluates the complete outcome of the agent's turn
- Training is consistent with rollout (both use end_features for values)
- Each decision uses the most relevant state information

=== PREVIOUS ISSUES FIXED ===

❌ BEFORE: Value stored using post_placement_node_features during rollout
✅ AFTER: Value stored using end_features during rollout

❌ BEFORE: PPO training used different features than rollout for values  
✅ AFTER: Both rollout and training use end_features for values

This ensures the value function learns to predict returns from the same states 
it evaluates during actual gameplay.
"""
