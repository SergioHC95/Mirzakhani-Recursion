from pathlib import Path
import wp

def main():
    ROOT = Path.cwd()
    cache_path = ROOT / "data" / "wp_rx.pkl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    rx = wp.iterate(dmax=200, cache_path=str(cache_path), max_workers=24)
    print(f"Database contains {len(rx)} intersection numbers.")

if __name__ == "__main__":
    main()
