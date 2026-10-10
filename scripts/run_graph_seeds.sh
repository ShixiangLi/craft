#!/usr/bin/env bash
# 顺序运行 seed 0..9；每次单独保存试验产物，不改写配置文件。
set -euo pipefail
trap 'exit 130' INT
trap 'exit 143' TERM

if (( $# > 1 )); then
    echo "用法: $0 [配置文件路径]" >&2
    exit 2
fi

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
config="${1:-$project_root/configs/graph.yaml}"
[[ "$config" = /* ]] || config="$PWD/$config"
python_bin="${GRAPH_PYTHON:-$project_root/.venv/bin/python}"
cd -- "$project_root"

if [[ ! -f "$config" || ! -x "$python_bin" ]]; then
    echo "请检查配置文件及 Python 路径: $config / $python_bin" >&2
    exit 2
fi

failed_seeds=()
for seed in {0..9}; do
    printf '\n开始试验 %d/10，seed=%d\n' "$((seed + 1))" "$seed"
    if "$python_bin" -m scripts.run_experiment --config "$config" --seed "$seed"; then
        printf 'seed=%d 运行完成\n' "$seed"
    else
        status=$?
        # 用户中断时停止整批；普通运行错误记录后继续下一 seed。
        if (( status == 130 || status == 143 )); then
            exit "$status"
        fi
        failed_seeds+=("$seed")
        printf 'seed=%d 运行出错（退出码 %d），继续下一次试验\n' "$seed" "$status" >&2
    fi
done

if (( ${#failed_seeds[@]} )); then
    printf '\n运行出错的 seeds: %s\n' "${failed_seeds[*]}" >&2
    exit 1
fi
printf '\n10 次试验均已运行完成，任务是否成功请查看各自的 summary.json。\n'
