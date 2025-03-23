from custom_dataset import QM9ProcessedDataset

dataset = QM9ProcessedDataset(root='../data/QM9Custom')

print(f"Loaded dataset with {len(dataset)} molecules")
data = dataset[0]
print(data)

