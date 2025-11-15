"""Experiment runner for GA vs SA comparison.

Runs GA and SA 100 times each and saves results to CSV.
CSV columns: run, algorithm, seed, best_f, total_delay, emergency_delay
"""
import csv
import random
import time
from ga import run_ga
from sa import run_sa
import ga as ga_module
import sa as sa_module
import statistics
import math
import matplotlib.pyplot as plt
try:
    import numpy as np
except Exception:
    np = None


def run_experiment(num_runs: int = 100):
    """Run GA and SA experiments and save results to CSV."""
    csv_filename = 'ga_sa_comparison_results.csv'
    
    print(f"Starting GA vs SA experiment ({num_runs} runs each)...")
    print(f"Results will be saved to: {csv_filename}\n")
    
    results = []
    
    # Run GA 100 times
    print("=" * 60)
    print("Running GA experiments...")
    print("=" * 60)
    for run_num in range(1, num_runs + 1):
        seed = run_num
        try:
            ga_best_perm, ga_best_speeds, ga_best_obj, _ = run_ga(
                pop_size=ga_module.POP_SIZE,
                generations=ga_module.GENERATIONS,
                random_seed=seed,
                return_history=False
            )
            
            results.append({
                'run': run_num,
                'algorithm': 'GA',
                'seed': seed,
                'best_f': ga_best_obj['f'],
                'total_delay': ga_best_obj['fall'],
                'emergency_delay': ga_best_obj['fem']
            })
            
            if run_num % 10 == 0:
                print(f"GA Run {run_num}/{num_runs} completed - f: {ga_best_obj['f']:.2f}")
        except Exception as e:
            print(f"GA Run {run_num} failed: {e}")
    
    print(f"\nGA experiments completed ({num_runs} runs)\n")
    
    # Run SA 100 times
    print("=" * 60)
    print("Running SA experiments...")
    print("=" * 60)
    for run_num in range(1, num_runs + 1):
        seed = run_num
        random.seed(seed)  # Seed for SA's random operations
        try:
            sa_best_perm, sa_best_speeds, sa_best_obj, _ = run_sa(
                T_init=sa_module.T_INITIAL,
                T_min=sa_module.T_MIN,
                cool_rate=sa_module.COOLING_RATE,
                iter_per_temp=sa_module.MAX_ITER_PER_TEMP,
                max_iter=sa_module.MAX_TOTAL_ITERATIONS,
                animation_enabled=False,
                return_history=False
            )
            
            # sa_best_obj is an objective dict with keys 'f','fall','fem'
            sa_best_f = sa_best_obj.get('f', float(sa_best_obj))
            sa_total = sa_best_obj.get('fall', 0.0)
            sa_em = sa_best_obj.get('fem', 0.0)
            results.append({
                'run': run_num,
                'algorithm': 'SA',
                'seed': seed,
                'best_f': sa_best_f,
                'total_delay': sa_total,
                'emergency_delay': sa_em
            })

            if run_num % 10 == 0:
                print(f"SA Run {run_num}/{num_runs} completed - f: {sa_best_f:.2f}")
        except Exception as e:
            print(f"SA Run {run_num} failed: {e}")
    
    print(f"\nSA experiments completed ({num_runs} runs)\n")
    
    # Save results to CSV
    print("=" * 60)
    print(f"Saving {len(results)} results to {csv_filename}...")
    print("=" * 60)
    
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['run', 'algorithm', 'seed', 'best_f', 'total_delay', 'emergency_delay']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    
    print(f"Results saved to: {csv_filename}\n")
    
    # Print summary statistics
    print("=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    ga_results = [r for r in results if r['algorithm'] == 'GA']
    sa_results = [r for r in results if r['algorithm'] == 'SA']
    
    if ga_results:
        ga_f_values = [r['best_f'] for r in ga_results]
        ga_delay_values = [r['total_delay'] for r in ga_results]
        ga_emergency_values = [r['emergency_delay'] for r in ga_results]
        
        print(f"\nGA Results (n={len(ga_results)}):")
        print(f"  Best f:")
        print(f"    Mean: {sum(ga_f_values) / len(ga_f_values):.2f}")
        print(f"    Std:  {(sum((x - sum(ga_f_values)/len(ga_f_values))**2 for x in ga_f_values) / len(ga_f_values))**0.5:.2f}")
        print(f"    Min:  {min(ga_f_values):.2f}")
        print(f"    Max:  {max(ga_f_values):.2f}")
        print(f"  Total Delay:")
        print(f"    Mean: {sum(ga_delay_values) / len(ga_delay_values):.2f}")
        print(f"    Std:  {(sum((x - sum(ga_delay_values)/len(ga_delay_values))**2 for x in ga_delay_values) / len(ga_delay_values))**0.5:.2f}")
        print(f"  Emergency Delay:")
        print(f"    Mean: {sum(ga_emergency_values) / len(ga_emergency_values):.2f}")
        print(f"    Std:  {(sum((x - sum(ga_emergency_values)/len(ga_emergency_values))**2 for x in ga_emergency_values) / len(ga_emergency_values))**0.5:.2f}")
    
    if sa_results:
        sa_f_values = [r['best_f'] for r in sa_results]
        
        print(f"\nSA Results (n={len(sa_results)}):")
        print(f"  Best f:")
        print(f"    Mean: {sum(sa_f_values) / len(sa_f_values):.2f}")
        print(f"    Std:  {(sum((x - sum(sa_f_values)/len(sa_f_values))**2 for x in sa_f_values) / len(sa_f_values))**0.5:.2f}")
        print(f"    Min:  {min(sa_f_values):.2f}")
        print(f"    Max:  {max(sa_f_values):.2f}")
    
    print("\n" + "=" * 60)
    print("Experiment Complete!")
    print("=" * 60)

    # Plot Gaussian distributions for GA and SA best objective (f)
    try:
        ga_values = [r['best_f'] for r in results if r['algorithm'] == 'GA']
        sa_values = [r['best_f'] for r in results if r['algorithm'] == 'SA']

        if ga_values and sa_values:
            ga_mean = statistics.mean(ga_values)
            ga_std = statistics.stdev(ga_values) if len(ga_values) > 1 else 0.0
            sa_mean = statistics.mean(sa_values)
            sa_std = statistics.stdev(sa_values) if len(sa_values) > 1 else 0.0

            # Determine x range
            vmin = min(min(ga_values), min(sa_values))
            vmax = max(max(ga_values), max(sa_values))
            margin = max(1.0, 0.1 * (vmax - vmin))
            x_min = vmin - margin
            x_max = vmax + margin

            if np is not None:
                xs = np.linspace(x_min, x_max, 400)
                def normal_pdf(x, mu, sigma):
                    if sigma <= 0:
                        return np.zeros_like(x)
                    coef = 1.0 / (sigma * math.sqrt(2 * math.pi))
                    return coef * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
                ga_pdf = normal_pdf(xs, ga_mean, ga_std)
                sa_pdf = normal_pdf(xs, sa_mean, sa_std)
            else:
                xs = [x_min + i * (x_max - x_min) / 399.0 for i in range(400)]
                def normal_pdf_scalar(x, mu, sigma):
                    if sigma <= 0:
                        return 0.0
                    coef = 1.0 / (sigma * math.sqrt(2 * math.pi))
                    return coef * math.exp(-0.5 * ((x - mu) / sigma) ** 2)
                ga_pdf = [normal_pdf_scalar(x, ga_mean, ga_std) for x in xs]
                sa_pdf = [normal_pdf_scalar(x, sa_mean, sa_std) for x in xs]

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.set_title('Best objective (f) distribution — GA vs SA')
            ax.hist(ga_values, bins=20, density=True, alpha=0.35, label=f'GA histogram (n={len(ga_values)})', color='C1')
            ax.hist(sa_values, bins=20, density=True, alpha=0.35, label=f'SA histogram (n={len(sa_values)})', color='C0')

            if np is not None:
                ax.plot(xs, ga_pdf, color='C1', lw=2, label=f'GA Normal PDF (μ={ga_mean:.2f}, σ={ga_std:.2f})')
                ax.plot(xs, sa_pdf, color='C0', lw=2, label=f'SA Normal PDF (μ={sa_mean:.2f}, σ={sa_std:.2f})')
            else:
                ax.plot(xs, ga_pdf, color='C1', lw=2, label=f'GA Normal PDF (μ={ga_mean:.2f}, σ={ga_std:.2f})')
                ax.plot(xs, sa_pdf, color='C0', lw=2, label=f'SA Normal PDF (μ={sa_mean:.2f}, σ={sa_std:.2f})')

            # Mean and ±1 std lines
            ax.axvline(ga_mean, color='C1', linestyle='--')
            ax.axvline(sa_mean, color='C0', linestyle='--')
            if ga_std > 0:
                ax.axvline(ga_mean - ga_std, color='C1', linestyle=':', alpha=0.6)
                ax.axvline(ga_mean + ga_std, color='C1', linestyle=':', alpha=0.6)
            if sa_std > 0:
                ax.axvline(sa_mean - sa_std, color='C0', linestyle=':', alpha=0.6)
                ax.axvline(sa_mean + sa_std, color='C0', linestyle=':', alpha=0.6)

            ax.set_xlabel('Best objective (f)')
            ax.set_ylabel('Density')
            ax.legend()
            plt.tight_layout()
            plt.show()
        else:
            print("Not enough data to plot GA/SA distributions.")
    except Exception as e:
        print(f"Failed to generate distribution plot: {e}")

if __name__ == '__main__':
    run_experiment(num_runs=100)
