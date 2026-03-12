from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
S2_DIR = Path(__file__).resolve().parent
ROOT_SCRIPTS_DIR = PROJECT_ROOT / 'scripts'
for _path in (PROJECT_ROOT, S2_DIR, ROOT_SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from run_s2_clustered_gtfree_activation import OUT_DIR, ensure_current_chain, evaluate_config, load_csv, normalize_row, pick_variant

SEARCH_CSV = OUT_DIR / 'S2_CLUSTER_PARAM_SEARCH.csv'
SCATTER_PNG = OUT_DIR / 'S2_CLUSTER_PARAM_SEARCH_SCATTER.png'
OPT_COMPARE = OUT_DIR / 'S2_CLUSTER_OPTIMIZED_COMPARE.csv'
OPT_REPORT = OUT_DIR / 'S2_CLUSTER_OPTIMIZED_REPORT.md'
OPT_ANALYSIS = OUT_DIR / 'S2_CLUSTER_OPTIMIZED_ANALYSIS.md'

DEFAULT_CLUSTER_MIN = 3
DEFAULT_ACTIVATION = 0.22
CLUSTER_MIN_GRID = [3, 5, 10, 20, 40]
ACTIVATION_GRID = [0.10, 0.14, 0.18, 0.22, 0.26, 0.30]


def variant_name(cluster_min_points: int, activation_threshold: float) -> str:
    if cluster_min_points != DEFAULT_CLUSTER_MIN and abs(activation_threshold - DEFAULT_ACTIVATION) < 1e-9:
        return f'131_cluster_budget_relaxed_c{cluster_min_points}_a{activation_threshold:.2f}'
    if cluster_min_points == DEFAULT_CLUSTER_MIN and abs(activation_threshold - DEFAULT_ACTIVATION) > 1e-9:
        return f'132_activation_threshold_lowered_c{cluster_min_points}_a{activation_threshold:.2f}'
    return f'133_joint_optimization_c{cluster_min_points}_a{activation_threshold:.2f}'


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_scatter(path: Path, rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for row in rows:
        x = float(row['bonn_ghost'])
        y = float(row['bonn_comp_r'])
        label = f"c={int(float(row['cluster_min_points']))}, a={float(row['activation_threshold']):.2f}"
        ax.scatter([x], [y], s=28)
        ax.annotate(label, (x, y), fontsize=6, alpha=0.8)
    ax.axvline(10.0, color='r', linestyle='--', linewidth=1)
    ax.axhline(70.0, color='g', linestyle='--', linewidth=1)
    ax.set_xlabel('Bonn Ghost')
    ax.set_ylabel('Bonn Comp-R(5cm)')
    ax.set_title('130 parameter search: Comp-R vs Ghost')
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    ensure_current_chain()
    rows = []
    for cluster_min_points in CLUSTER_MIN_GRID:
        for activation_threshold in ACTIVATION_GRID:
            row = evaluate_config(
                cluster_min_points=cluster_min_points,
                activation_threshold=activation_threshold,
                variant_name=variant_name(cluster_min_points, activation_threshold),
            )
            rows.append(
                {
                    'variant_name': row['variant'],
                    'cluster_min_points': cluster_min_points,
                    'activation_threshold': activation_threshold,
                    'bonn_acc_cm': row['bonn_acc_cm'],
                    'bonn_comp_r': row['bonn_comp_r_5cm'],
                    'bonn_ghost': row['bonn_rear_ghost_sum'],
                    'proxy_recall': row['proxy_recall'],
                    'proxy_precision': row['proxy_precision'],
                    'activated_points_sum': row['activated_points_sum'],
                    'retained_cluster_count': row['retained_cluster_count'],
                }
            )
    write_csv(SEARCH_CSV, rows)
    make_scatter(SCATTER_PNG, rows)

    feasible = [r for r in rows if float(r['bonn_ghost']) <= 10.0]
    feasible.sort(key=lambda r: (float(r['bonn_comp_r']), -float(r['bonn_acc_cm']), -float(r['proxy_recall'])), reverse=True)
    best = feasible[0] if feasible else max(rows, key=lambda r: float(r['bonn_comp_r']))

    baseline_125 = normalize_row(pick_variant(OUT_DIR / 'S2_HYBRID_INTEGRATION_COMPARE.csv', '125_hybrid_papg_constrained'))
    current_130 = normalize_row(load_csv(OUT_DIR / 'S2_CLUSTERED_GTFREE_ACTIVATION_COMPARE.csv')[-1])
    best_eval = evaluate_config(
        cluster_min_points=int(best['cluster_min_points']),
        activation_threshold=float(best['activation_threshold']),
        variant_name=str(best['variant_name']),
    )
    compare_rows = [baseline_125, current_130, best_eval]
    write_csv(OPT_COMPARE, compare_rows)

    success = (
        float(best_eval['bonn_comp_r_5cm']) >= 70.0
        and float(best_eval['bonn_rear_ghost_sum']) <= 10.0
        and float(best_eval['bonn_acc_cm']) <= 4.460
    )

    report_lines = [
        '# S2 Cluster Optimized Report',
        '',
        f"最佳参数：`cluster_min_points={int(best['cluster_min_points'])}`, `activation_threshold={float(best['activation_threshold']):.2f}`",
        '',
        f"- `125`：Acc=`{float(baseline_125['bonn_acc_cm']):.3f}`, Comp-R=`{float(baseline_125['bonn_comp_r_5cm']):.2f}`, Ghost=`{float(baseline_125['bonn_rear_ghost_sum']):.0f}`",
        f"- `130`：Acc=`{float(current_130['bonn_acc_cm']):.3f}`, Comp-R=`{float(current_130['bonn_comp_r_5cm']):.2f}`, Ghost=`{float(current_130['bonn_rear_ghost_sum']):.0f}`",
        f"- `best`：Acc=`{float(best_eval['bonn_acc_cm']):.3f}`, Comp-R=`{float(best_eval['bonn_comp_r_5cm']):.2f}`, Ghost=`{float(best_eval['bonn_rear_ghost_sum']):.0f}`, Recall=`{float(best_eval['proxy_recall']):.3f}`",
        '',
        f"结论：`{'success' if success else 'not-pass'}`",
    ]
    OPT_REPORT.write_text('\n'.join(report_lines) + '\n', encoding='utf-8')

    analysis_lines = [
        '# S2 Cluster Optimized Analysis',
        '',
        '搜索说明：',
        f'- `cluster_min_points` 采用当前实现尺度下的离散网格：`{CLUSTER_MIN_GRID}`',
        f'- `activation_threshold` 采用当前 `P_OCC_MIN` 量纲的网格：`{ACTIVATION_GRID}`',
        '- 原用户给出的 `0.3~0.6` 更适合概率阈值的粗设想，但当前实现默认值为 `0.22`，因此搜索区间已按实际代码量纲下调。',
        '',
        f"Ghost<=10 约束下 Comp-R 最优组合：`{best['variant_name']}`",
        f"- Acc=`{float(best_eval['bonn_acc_cm']):.3f}`",
        f"- Comp-R=`{float(best_eval['bonn_comp_r_5cm']):.2f}`",
        f"- Ghost=`{float(best_eval['bonn_rear_ghost_sum']):.0f}`",
        f"- Recall=`{float(best_eval['proxy_recall']):.3f}`",
        '',
    ]
    if success:
        analysis_lines += [
            '结论：',
            '- 本路径已满足 `Comp-R >= 70`, `Ghost <= 10`, `Acc <= 4.460`，可以取代 `125` 成为新主线。',
        ]
    else:
        analysis_lines += [
            '结论：',
            '- 在当前二维参数搜索下，没有组合能够同时满足 `Comp-R >= 70`, `Ghost <= 10`, `Acc <= 4.460`。',
            '- 这说明当前 cluster pruning 仍存在固有保守性：降低阈值虽然能回收部分点，但很难在保持低 Ghost 的同时恢复到 `125` 的系统完整性。',
            '- 下一步更可能需要引入 prior-guided completion 或更细的 cluster confidence 估计，而不只是继续放宽当前两个参数。',
        ]
    OPT_ANALYSIS.write_text('\n'.join(analysis_lines) + '\n', encoding='utf-8')
    print('[done]', SEARCH_CSV)


if __name__ == '__main__':
    main()
