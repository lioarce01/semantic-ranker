#!/usr/bin/env python3
"""
Monitor QG-Rerank training for overfitting and issues
Run this alongside training to detect problems early
"""

import time
import re
import subprocess
import sys
from pathlib import Path

def monitor_training():
    """Monitor training output for overfitting signs"""

    print("🔍 QG-Rerank Training Monitor")
    print("=" * 50)
    print("Monitoring for overfitting and training issues...")
    print("Press Ctrl+C to stop monitoring\n")

    # Track metrics over time
    ndcg_history = []
    mrr_history = []
    bce_history = []
    contrastive_history = []
    rank_history = []

    try:
        # Read from stdin (training output)
        for line in sys.stdin:
            line = line.strip()

            # Extract NDCG@10
            ndcg_match = re.search(r'NDCG@10: ([0-9.]+)', line)
            if ndcg_match:
                ndcg = float(ndcg_match.group(1))
                ndcg_history.append(ndcg)

                if ndcg > 0.95 and len(ndcg_history) > 2:
                    print(f"\n⚠️ WARNING: Very high NDCG@10 ({ndcg:.4f}) - possible overfitting!")
                    print("💡 Consider: reducing learning rate, increasing dropout, limiting graph size"
                elif ndcg > 0.99:
                    print(f"\n🚨 CRITICAL: Extremely high NDCG@10 ({ndcg:.4f}) - severe overfitting!")
                    print("💡 IMMEDIATE ACTION: Stop training, reduce model capacity"
            # Extract MRR@10
            mrr_match = re.search(r'MRR@10: ([0-9.]+)', line)
            if mrr_match:
                mrr = float(mrr_match.group(1))
                mrr_history.append(mrr)

            # Extract losses
            bce_match = re.search(r'BCE: ([0-9.]+)', line)
            if bce_match:
                bce = float(bce_match.group(1))
                bce_history.append(bce)

            contrastive_match = re.search(r'Contrastive: ([0-9.]+)', line)
            if contrastive_match:
                contrastive = float(contrastive_match.group(1))
                contrastive_history.append(contrastive)

                if contrastive == 0.0 and len(contrastive_history) > 3:
                    print(f"\n⚠️ WARNING: Contrastive loss is 0.0000 - GNN not working!")

            rank_match = re.search(r'Rank: ([0-9.]+)', line)
            if rank_match:
                rank = float(rank_match.group(1))
                rank_history.append(rank)

                if rank == 0.0 and len(rank_history) > 3:
                    print(f"\n⚠️ WARNING: Rank loss is 0.0000 - GNN ranking not working!")

            # Print current status every 10 lines
            if len(ndcg_history) % 5 == 0 and ndcg_history:
                print(f"📊 Status: NDCG@10={ndcg_history[-1]:.4f}, "
                      f"BCE={bce_history[-1]:.4f}, "
                      f"Contrastive={contrastive_history[-1]:.4f}, "
                      f"Rank={rank_history[-1]:.4f}")

            # Detect overfitting trend
            if len(ndcg_history) >= 3:
                recent_ndcg = ndcg_history[-3:]
                if all(x > 0.9 for x in recent_ndcg) and len(set(recent_ndcg)) == 1:
                    print(f"\n🚨 OVERFITTING DETECTED: NDCG@10 stuck at {recent_ndcg[0]:.4f}")
                    print("💡 Solutions: early stopping, regularization, data augmentation")

    except KeyboardInterrupt:
        print("\n📊 Final Statistics:")
        if ndcg_history:
            print(f"  Final NDCG@10: {ndcg_history[-1]:.4f}")
            print(f"  Peak NDCG@10: {max(ndcg_history):.4f}")
            print(f"  Evaluations: {len(ndcg_history)}")

        print("Monitoring stopped.")

if __name__ == "__main__":
    print("Usage: python scripts/monitor_qg_training.py < training_output.log")
    print("Or:    python -m cli.qg_train [args] 2>&1 | python scripts/monitor_qg_training.py")
    print()
    monitor_training()