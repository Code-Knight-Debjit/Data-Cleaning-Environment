import sys
sys.path.insert(0, "data_cleaning_env")

from server.data_cleaning_env import DataCleaningEnvironment
from models import CleanAction

import random

env = DataCleaningEnvironment()
obs = env.reset(task_id="medium")

columns = ["price", "quantity", "order_date", "customer", "region"]
total_reward = 0
step = 0

while not obs.done and step < 100:
    print(f"Step {step}  Current score: {obs.current_score:.4f}  Reward: {obs.reward}")
    # print(f"Observation: {obs}")
    print("-" * 50)
    # Random action (poor policy — just for testing reward signal)
    cmd = random.choice(["SET_VALUE", "FILL_MISSING", "STANDARDIZE_COL"])
    if cmd == "SET_VALUE":
        action = CleanAction(command="SET_VALUE", row_index=random.randint(0,49),
                             column=random.choice(columns), value="10.0")
    elif cmd == "FILL_MISSING":
        action = CleanAction(command="FILL_MISSING",
                             column=random.choice(["price", "quantity"]),
                             fill_strategy="median")
    else:
        action = CleanAction(command="STANDARDIZE_COL",
                             column=random.choice(columns))

    obs = env.step(action)
    total_reward += obs.reward or 0
    step += 1

print(f"Episode done. Steps={step}  Total reward={total_reward:.3f}  Final score={obs.current_score:.4f}")