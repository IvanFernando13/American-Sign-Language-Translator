import os
import pandas as pd

# Path to your training folder
TRAIN_DIR = "dataset/Train_Alphabet"

def check_dataset_balance():
    if not os.path.exists(TRAIN_DIR):
        print(f"❌ Error: {TRAIN_DIR} directory not found.")
        return

    stats = []
    folders = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
    
    for folder in folders:
        path = os.path.join(TRAIN_DIR, folder)
        # Count only image files
        count = len([f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        stats.append({'Class': folder, 'Count': count})
    
    df = pd.DataFrame(stats)
    avg_count = df['Count'].mean()
    
    # Identify under-represented classes
    df['Status'] = df['Count'].apply(lambda x: 'OK' if x >= avg_count * 0.8 else '⚠️ LOW DATA')
    
    print("\n--- Dataset Balance Report ---")
    print(df.to_string(index=False))
    print(f"\nAverage Images per Class: {int(avg_count)}")
    
    # Specific warnings for your problem letters
    problem_letters = ['P', 'R', 'V', 'T', 'W', 'X', 'Y']
    low_priority = df[(df['Status'] == '⚠️ LOW DATA') & (df['Class'].str.contains('|'.join(problem_letters)))]
    
    if not low_priority.empty:
        print("\n🔥 URGENT ACTION REQUIRED:")
        print(f"Add more images for these difficult letters: {low_priority['Class'].tolist()}")
    else:
        print("\n✅ Problem letters have sufficient data relative to the average.")

if __name__ == "__main__":
    check_dataset_balance()