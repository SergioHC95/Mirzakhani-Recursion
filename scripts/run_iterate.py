from pathlib import Path
import wp
import argparse

def main(max_workers):
    ROOT = Path.cwd()
    cache_path = ROOT / "data" / "wp_rx.pkl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    rx = wp.iterate(dmax=200, cache_path=str(cache_path), max_workers=max_workers)
    print(f"Database contains {len(rx)} intersection numbers.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_workers', type=int, default=6,
                        help='Number of worker processes (default: 6)')
    args = parser.parse_args()
    main(args.max_workers)
