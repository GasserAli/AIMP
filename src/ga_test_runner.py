import traceback
from ga import run_ga

if __name__ == '__main__':
    try:
        run_ga(pop_size=30, generations=30, random_seed=2)
    except Exception:
        traceback.print_exc()
