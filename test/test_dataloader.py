import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from models.data_utils import load_data, create_dataloader

# Load data
dataset, vocab = load_data('data/train_data_combined.txt', vocab_size=5000, max_seq_length=32, min_freq=2)

# Create dataloader
dataloader = create_dataloader(dataset, batch_size=8, shuffle=True)

print('\nDataLoader created:')
print(f'  Batches: {len(dataloader)}')
print('  First batch: ')
first_batch = next(iter(dataloader))
print(f'    Input IDs shape: {first_batch["input_ids"].shape}')
print(f'    Target IDs shape: {first_batch["target_ids"].shape}')

# Wait for user to press Enter so output can be reviewed interactively
try:
	input('\nPress Enter to exit...')
except Exception:
	pass
