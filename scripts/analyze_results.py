"""离线生成已完成实验的图表：python -m scripts.analyze_results --run-dir ..."""
import argparse
from utils.visualization import visualize_run


def main() -> None:
    parser = argparse.ArgumentParser(description='从已有实验记录生成 PNG / SVG 图表，不重新运行实验')
    parser.add_argument('--run-dir', required=True, help='包含 summary.json 的运行目录')
    args = parser.parse_args()
    for path in visualize_run(args.run_dir):
        print(path)


if __name__ == '__main__':
    main()
