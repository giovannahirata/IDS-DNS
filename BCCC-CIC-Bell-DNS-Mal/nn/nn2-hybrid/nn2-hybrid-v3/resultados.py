import pickle
import pandas as pd

with open('results_benchmark.pkl', 'rb') as f:
	results = pickle.load(f)

df_benchmark = pd.DataFrame(results['benchmark'])

print("\nResultados\n")
print(df_benchmark.to_string())

print("\nResultados do ensemble (média)\n")
print(f"Acurácia: {results['ensemble_accuracy']:.4f}")
print(f"Precisão: {results['ensemble_precision']:.4f}")
print(f"Recall: {results['ensemble_recall']:.4f}")
print(f"F1-Score: {results['ensemble_f1']:.4f}")
print(f"AUC-ROC: {results['ensemble_auc']:.4f}")

print(f"\nMelhor modelo individual: {results['best_model']}")
