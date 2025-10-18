import json
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# Load the dataset
with open('HateXplain/Data/dataset.json', 'r') as f:
    data = json.load(f)

# Initialize counters
label_counter = Counter()
target_counter = Counter()

# Process each post
for post_id, post_data in data.items():
    # Get labels from all annotators
    for annotator in post_data.get('annotators', []):
        label = annotator.get('label', '')
        if label:
            label_counter[label] += 1

        # Get targets (offensive types)
        targets = annotator.get('target', [])
        for target in targets:
            if target and target != 'None':
                target_counter[target] += 1

# Print statistics
print("\n=== Dataset Statistics ===")
print(f"Total posts: {len(data)}")
print(f"\nLabel distribution (from all annotations):")
for label, count in label_counter.most_common():
    print(f"  {label}: {count}")

print(f"\nOffensive type distribution:")
for target, count in target_counter.most_common():
    print(f"  {target}: {count}")

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Pie chart 1: Label distribution
labels1 = list(label_counter.keys())
sizes1 = list(label_counter.values())
colors1 = ['#ff6b6b', '#4ecdc4', '#45b7d1'][:len(labels1)]

# Calculate percentages
total1 = sum(sizes1)
percentages1 = [size/total1 * 100 for size in sizes1]

# Create labels with counts and percentages
labels1_with_counts = [f'{label}\n({count:,} - {pct:.1f}%)'
                       for label, count, pct in zip(labels1, sizes1, percentages1)]

wedges1, texts1, autotexts1 = ax1.pie(sizes1, labels=labels1_with_counts, colors=colors1,
                                       autopct='', startangle=90)
ax1.set_title('Distribution of Labels in HateXplain Dataset', fontsize=14, fontweight='bold')

# Pie chart 2: Offensive type distribution (top 10)
top_targets = target_counter.most_common(10)
labels2 = [target for target, _ in top_targets]
sizes2 = [count for _, count in top_targets]

# Use a colormap for more colors
colors2 = plt.cm.Set3(np.linspace(0, 1, len(labels2)))

# Calculate percentages
total2 = sum(sizes2)
percentages2 = [size/total2 * 100 for size in sizes2]

# Create labels with counts and percentages
labels2_with_counts = [f'{label}\n({count:,} - {pct:.1f}%)'
                       for label, count, pct in zip(labels2, sizes2, percentages2)]

wedges2, texts2, autotexts2 = ax2.pie(sizes2, labels=labels2_with_counts, colors=colors2,
                                       autopct='', startangle=90)
ax2.set_title('Distribution of Offensive Types (Top 10)', fontsize=14, fontweight='bold')

# Adjust text properties for better readability
for text in texts1 + texts2:
    text.set_fontsize(9)

plt.suptitle('HateXplain Dataset Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()

# Save the figure
plt.savefig('hatexplain_distribution_charts.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Pie charts saved as 'hatexplain_distribution_charts.png'")

plt.show()