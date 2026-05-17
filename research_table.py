import pandas as pd

# Assuming you already ran the classification_report logic from the previous script
report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)

# Convert dictionary to a DataFrame for research formatting
df_report = pd.DataFrame(report).transpose()

# Remove 'accuracy', 'macro avg', and 'weighted avg' for the per-class table
per_class_table = df_report.iloc[:-3, :].copy()

# Format for the paper: Round to 3 decimal places
per_class_table = per_class_table[['precision', 'recall', 'f1-score']].round(3)

print("\n--- RESEARCH SUMMARY TABLE (Copy into Word/LaTeX) ---")
print(per_class_table.to_markdown()) # Use .to_latex() if you are using Overleaf/LaTeX

# Save to CSV so you can open it in Excel to make charts
per_class_table.to_csv("asl_research_results.csv")
print("\n✅ Results saved to 'asl_research_results.csv'")