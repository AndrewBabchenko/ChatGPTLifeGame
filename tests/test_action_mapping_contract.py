"""
Contract test: Ensures action mapping matches Animal._apply_action() exactly
This prevents regression where direction→action mapping gets out of sync with the environment
"""
from tests.test_directional_pretraining_learns import best_action_for


def test_action_mapping_matches_env_contract():
    """
    Environment contract (from Animal._apply_action):
    0: (0,-1) N
    1: (1,-1) NE  
    2: (1, 0) E
    3: (1, 1) SE
    4: (0, 1) S
    5: (-1,1) SW
    6: (-1,0) W
    7: (-1,-1) NW
    """
    assert best_action_for(0, -1, device=None) == 0, "North mapping broken"
    assert best_action_for(1, -1, device=None) == 1, "NorthEast mapping broken"
    assert best_action_for(1,  0, device=None) == 2, "East mapping broken"
    assert best_action_for(1,  1, device=None) == 3, "SouthEast mapping broken"
    assert best_action_for(0,  1, device=None) == 4, "South mapping broken"
    assert best_action_for(-1, 1, device=None) == 5, "SouthWest mapping broken"
    assert best_action_for(-1, 0, device=None) == 6, "West mapping broken"
    assert best_action_for(-1,-1, device=None) == 7, "NorthWest mapping broken"
    
    print("✓ Action mapping contract validated: all 8 directions match environment")


if __name__ == "__main__":
    test_action_mapping_matches_env_contract()
