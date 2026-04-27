import asyncio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from veritas.db.veritas_db import db
from scripts.stats.common import COLORS, _ensure_plots_dir
import os
from pathlib import Path

async def fetch_and_compute_similarity(kind: str):
    print(f"Fetching {kind} embeddings...")
    rows = await db.get_all_embeddings(kind)
    if not rows:
        print(f"No embeddings found for kind: {kind}")
        return None, None

    ids = [row[0] for row in rows]
    embeddings = np.array([row[1] for row in rows])
    
    print(f"Computing pairwise cosine similarity for {len(embeddings)} {kind} embeddings...")
    # cosine_similarity computes the similarity between all pairs in X and Y.
    # If Y is None, it computes similarity between all pairs in X.
    sim_matrix = cosine_similarity(embeddings)
    
    # We only need the upper triangle (excluding the diagonal)
    # as the matrix is symmetric and the diagonal is always 1.
    iu = np.triu_indices(sim_matrix.shape[0], k=1)
    similarities = sim_matrix[iu]
    
    return ids, similarities, iu

def plot_histogram(similarities, kind, output_dir):
    if similarities is None or len(similarities) == 0:
        return

    plt.figure(figsize=(10, 6))
    plt.hist(similarities, bins=50, color=COLORS.get("blue", "blue"), edgecolor='black', alpha=0.7)
    plt.yscale('log')
    plt.title(f"Distribution of Pairwise Cosine Similarities ({kind})")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    output_path = output_dir / f"cosine_similarity_hist_{kind}.pdf"
    plt.savefig(output_path)
    plt.show()
    plt.close()
    print(f"Saved histogram: {output_path}")

def report_stats(ids, similarities, iu, kind):
    if similarities is None or len(similarities) == 0:
        return

    print(f"\nStatistics for {kind} embeddings:")
    print(f"  Count: {len(similarities)} pairs")
    print(f"  Mean:   {np.mean(similarities):.4f}")
    print(f"  Median: {np.median(similarities):.4f}")
    print(f"  Min:    {np.min(similarities):.4f}")
    print(f"  Max:    {np.max(similarities):.4f}")
    print(f"  Std:    {np.std(similarities):.4f}")
    
    # Top 10 most similar pairs
    print(f"\n  Top 10 most similar {kind} pairs:")
    top_indices = np.argsort(similarities)[::-1][:10]
    for idx in top_indices:
        i, j = iu[0][idx], iu[1][idx]
        print(f"    IDs ({ids[i]}, {ids[j]}): {similarities[idx]:.4f}")

    # Thresholds for "duplicates"
    for threshold in [0.8, 0.9, 0.95, 0.99]:
        mask = similarities >= threshold
        count = np.sum(mask)
        percentage = (count / len(similarities)) * 100
        print(f"\n  Pairs with similarity >= {threshold}: {count} ({percentage:.2f}%)")
        
        if count > 0:
            print(f"    10 least similar pairs above {threshold}:")
            # Get indices of elements that satisfy the mask
            mask_indices = np.where(mask)[0]
            # Get similarities of these elements
            threshold_sims = similarities[mask_indices]
            # Sort them and take the first 10
            least_sim_indices_in_mask = np.argsort(threshold_sims)[:10]
            # Map back to original similarities indices
            final_indices = mask_indices[least_sim_indices_in_mask]
            
            for idx in final_indices:
                i, j = iu[0][idx], iu[1][idx]
                print(f"      IDs ({ids[i]}, {ids[j]}): {similarities[idx]:.4f}")

async def main():
    await db.connect_maybe_initialize()
    
    output_dir = Path(_ensure_plots_dir()) / "similarity_stats"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    kinds = ["text", "image", "video"]
    
    for kind in kinds:
        ids, similarities, iu = await fetch_and_compute_similarity(kind)
        if similarities is not None:
            report_stats(ids, similarities, iu, kind)
            plot_histogram(similarities, kind, output_dir)
        else:
            print(f"Skipping {kind} as no data was found.")

if __name__ == "__main__":
    asyncio.run(main())
