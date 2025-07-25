import json
from torch.utils.tensorboard import SummaryWriter

learning_file = open(f"../../learning_stats_2025-07-24_09-26-48.txt", "r")
writer = SummaryWriter(log_dir="logs/Atilla_World_extra_aggressive_better_balanced")  # Store data here

wins = 0
for i, line in enumerate(learning_file.readlines()):
    if i == 0:
        continue
    js = json.loads(line)
    if js['win']:
        wins += 1
    writer.add_scalar("win_rate", wins/i, i)
    writer.add_scalar("pred_value vs value", js['value_mean'], js['value_pred_mean'])
    writer.add_scalar("pred_value over value", js['value_mean']/js['value_pred_mean'], i )

    for key, values in js.items():
        writer.add_scalar(key, values, i)
