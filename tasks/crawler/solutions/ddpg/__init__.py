import os
SOLUTIONS_CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'solution_checkpoint')
if not os.path.exists(SOLUTIONS_CHECKPOINT_DIR):
    os.makedirs(SOLUTIONS_CHECKPOINT_DIR)
