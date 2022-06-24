The package `pursuit_game` is a part of PettingZoo-v1.9.0 (https://github.com/Farama-Foundation/PettingZoo/tree/1.9.0).
Some important modifications are recorded in the following.

Important modifications:
- Replace all `import pettingzoo.utils` with `import pettingzoo_utils`.
  In this way, the package `pursuit_game` is self-contained and no dependicies on the pettingzoo package.
- Add the function `def last_vsos(self, agent_id, observe=True)` in `pursuit_game/pettingzoo_utils/env.py`.
- `pursuit_game/utils/agent_utils.py` is replaced `pursuit_game/utils/agent_utils_vsop.py`.
- `pursuit_game/utils/simple_graph_vsos.py` is newly added.
- `pursuit_game/utils/discrete_agent.py` `def step(self, a):` is modified.

Modifications in pursuit_vsos.py and pursuit_base_vsos.py:
- Reward structure.
- Initialization.

Strategy to deal with out-of-boundary agent is written in `pursuit_game/utils/discrete_agent.py` `def step(self, a):`,
where out-of-boundary agent can
- either die immediately in its latest legal position,
- or keep alive yet stay in its latest legal position.
- a property `collide_with_obstacle` is added to store this information.

Calling Relationship:
- `pursuit_game/utils/discrete_agent.py`: define `class DiscreteAgent` with `def step(self, a):`.
- `pursuit_game/utils/agent_utils_vsop.py` call `pursuit_game/utils/discrete_agent.py` `class DiscreteAgent`
  to initialize agents (pursuers and evaders).
- `pursuit_game/pursuit_base_vsos.py`:
    -- call `pursuit_game/utils/agent_utils_vsop.py` to initialize `self.pursuers` and 'self.evaders`.
    -- And `self.pursuers` and 'self.evaders` are used to create `self.pursuer_layer` and `self.evader_layer`,
       respectively.
    -- in `def step(self, action, agent_id, is_last):`, `agent_layer.move_agent(agent_id, action)`
       call `pursuit_game/utils/agent_layer.py` `def move_agent(self, agent_idx, action)`,
       which calls `pursuit_game/utils/discrete_agent.py` `def step(self, a):`.

Extract and modify part of the SuperSuit 3.0.1 codes as the package `pursuit_game/supersuit`.

Observation.
Pettingzoo/sisl/pursuit/pursuit.py, def observe(agent):, call
pettingzoo/sisl/pursuit/pursuit_base.py, def safely_observe():, call
def collect_obs(agent_layer, i):, call
def collect_obs_by_idx(agent_layer, agent_idx):  return …
- Format: numpy array, float32, 3 x obs_range x obs_range, default: 0.0.
0: border walls: 1.0; centered obstacle: -1.0.
1: pursuers, value is the number of pursuers.
2: evaders, value is the number of evaders.
Location: pursuit_base.py

Actions encoding.
- Location: pettingzoo/sisl/pursuit/utils/discrete_agent.py
- Coordination systems.
- Origin locates in the upper left.
- Horizontal axis of the 2-D map is the first dimension while vertical axis is the second dimension.
- Default action encoding.
    self.eactions = [0,  # move left
                     1,  # move right
                     2,  # move up
                     3,  # move down
                     4]  # stay
