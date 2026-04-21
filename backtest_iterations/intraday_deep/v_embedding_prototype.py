#!/usr/bin/env python3
"""
PROTOTYPE: Can text embeddings of event descriptions predict bounce success?

Uses the 336 labeled events that have stock_event_description text.
Embeds the text, combines with numeric features, trains a classifier,
and compares accuracy with vs without embeddings.
"""
import pandas as pd
import numpy as np
import os, sys, warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load events with descriptions
events = pd.read_csv('outputs/events_fully_labeled.csv')
events['event_date'] = pd.to_datetime(events['event_date'])

# Filter to events that have text descriptions
labeled = events[events['stock_event_description'].notna() & (events['stock_event_description'] != '')].copy()
print(f"Events with text descriptions: {len(labeled)}")

# Also include events with event type labels (even without description)
# For those, we'll create a synthetic description from the structured fields
has_type = events[events['stock_event_type'].notna() & ~events['stock_event_type'].isin(['unlabeled', ''])].copy()
print(f"Events with event type labels: {len(has_type)}")

# Build text descriptions for all labeled events
def build_description(row):
    parts = []
    if pd.notna(row.get('stock_event_description')) and row['stock_event_description'] != '':
        parts.append(str(row['stock_event_description']))
    
    # Add structured context
    parts.append(f"{row['ticker']} dropped {row['drop_pct']*100:.1f}% on {row['event_date'].date()}")
    
    if pd.notna(row.get('stock_event_type')) and row['stock_event_type'] not in ['unlabeled', '']:
        parts.append(f"Event type: {row['stock_event_type']}, severity: {row.get('stock_event_severity', 'unknown')}")
    
    if pd.notna(row.get('sector_etf')):
        parts.append(f"Sector: {row['sector_etf']}")
    
    stress = row.get('liquidity_credit_stress_severity', 'none')
    if stress != 'none':
        parts.append(f"Market stress: {stress}")
    
    rotation = row.get('sector_rotation_direction', '')
    if rotation and rotation != 'none':
        parts.append(f"Sector rotation: {rotation}")
    
    return '. '.join(parts)

has_type['text'] = has_type.apply(build_description, axis=1)
print(f"\nSample text:")
print(has_type['text'].iloc[0][:200])
print("...")
print(has_type['text'].iloc[5][:200])

# Build numeric features
numeric_cols = ['drop_pct']
# Add binary features for stress, rotation, etc.
has_type['stress_none'] = (has_type['liquidity_credit_stress_severity'] == 'none').astype(int)
has_type['stress_low'] = (has_type['liquidity_credit_stress_severity'] == 'low').astype(int)
has_type['stress_medium'] = (has_type['liquidity_credit_stress_severity'] == 'medium').astype(int)
has_type['stress_high'] = (has_type['liquidity_credit_stress_severity'] == 'high').astype(int)
has_type['outflow'] = (has_type['sector_rotation_direction'] == 'outflow').astype(int)
has_type['friday'] = (has_type['event_date'].dt.dayofweek == 4).astype(int)

numeric_features = ['drop_pct', 'stress_none', 'stress_low', 'stress_medium', 'stress_high', 'outflow', 'friday']

# Target
has_type['target'] = has_type['success'].astype(int)

print(f"\nDataset: {len(has_type)} events, {has_type['target'].mean()*100:.1f}% success rate")

# Embed the text
print("\nLoading embedding model (this may take a minute on first run)...")
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim, fast, free
print("Embedding texts...")
embeddings = model.encode(has_type['text'].tolist(), show_progress_bar=True, batch_size=64)
print(f"Embedding shape: {embeddings.shape}")

# Build feature matrices
X_numeric = has_type[numeric_features].values
X_embed = embeddings
X_combined = np.hstack([X_numeric, X_embed])
y = has_type['target'].values

print(f"\nFeature dimensions:")
print(f"  Numeric only: {X_numeric.shape}")
print(f"  Embeddings only: {X_embed.shape}")
print(f"  Combined: {X_combined.shape}")

# Train and evaluate with cross-validation
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"\n{'='*70}")
print(f"  CROSS-VALIDATED ACCURACY (5-fold)")
print(f"{'='*70}\n")

results = {}

for name, X in [("Numeric only", X_numeric), ("Embeddings only", X_embed), ("Combined", X_combined)]:
    for model_name, clf in [
        ("Logistic Regression", Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, random_state=42))])),
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)),
        ("Gradient Boosting", GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=3)),
    ]:
        scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
        auc_scores = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
        key = f"{name} + {model_name}"
        results[key] = {'accuracy': scores.mean(), 'auc': auc_scores.mean()}
        print(f"  {key:>45}: acc={scores.mean():.3f} (±{scores.std():.3f}) | AUC={auc_scores.mean():.3f}")

# Baseline: always predict majority class
majority_acc = max(y.mean(), 1 - y.mean())
print(f"\n  {'Baseline (always predict majority)':>45}: acc={majority_acc:.3f}")

# Summary
print(f"\n{'='*70}")
print(f"  SUMMARY: Does text help?")
print(f"{'='*70}\n")

best_numeric = max(v['auc'] for k, v in results.items() if 'Numeric only' in k)
best_embed = max(v['auc'] for k, v in results.items() if 'Embeddings only' in k)
best_combined = max(v['auc'] for k, v in results.items() if 'Combined' in k)

print(f"  Best AUC with numeric only:    {best_numeric:.3f}")
print(f"  Best AUC with embeddings only: {best_embed:.3f}")
print(f"  Best AUC with combined:        {best_combined:.3f}")
print(f"  Improvement from adding text:  {best_combined - best_numeric:+.3f}")

if best_combined > best_numeric + 0.01:
    print(f"\n  → YES, text embeddings add predictive power (+{(best_combined-best_numeric)*100:.1f}% AUC)")
elif best_combined > best_numeric:
    print(f"\n  → MARGINAL improvement from text ({(best_combined-best_numeric)*100:.1f}% AUC)")
else:
    print(f"\n  → NO, text doesn't help with this sample size")

# Feature importance from the best combined model
print(f"\n{'='*70}")
print(f"  WHAT THE MODEL LEARNED")
print(f"{'='*70}\n")

# Train on full data to inspect
rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
rf.fit(X_combined, y)

# Feature importances
importances = rf.feature_importances_
numeric_importance = importances[:len(numeric_features)].sum()
embedding_importance = importances[len(numeric_features):].sum()

print(f"  Feature importance split:")
print(f"    Numeric features: {numeric_importance*100:.1f}%")
print(f"    Text embeddings:  {embedding_importance*100:.1f}%")
print()
print(f"  Top numeric features:")
for i, col in enumerate(numeric_features):
    print(f"    {col:>20}: {importances[i]*100:.2f}%")

# Save
with open(os.path.join(OUT_DIR, 'v_embedding_results.txt'), 'w') as f:
    f.write("EMBEDDING PROTOTYPE RESULTS\n\n")
    f.write(f"Events: {len(has_type)}\n")
    f.write(f"Success rate: {has_type['target'].mean()*100:.1f}%\n\n")
    f.write(f"Best AUC numeric: {best_numeric:.3f}\n")
    f.write(f"Best AUC embeddings: {best_embed:.3f}\n")
    f.write(f"Best AUC combined: {best_combined:.3f}\n")
    f.write(f"Improvement: {best_combined - best_numeric:+.3f}\n\n")
    f.write(f"Numeric importance: {numeric_importance*100:.1f}%\n")
    f.write(f"Embedding importance: {embedding_importance*100:.1f}%\n")

print(f"\nSaved to {OUT_DIR}/v_embedding_results.txt")
