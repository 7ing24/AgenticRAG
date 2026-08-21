#!/usr/bin/env python
"""RAG 评测入口脚本

用法:
    python3 eval/run_eval.py --mode full --load synthetic_qa.json               # 单次评测
    python3 eval/run_eval.py --mode baseline --load synthetic_qa.json           # baseline 模式
    python3 eval/run_eval.py --mode react --load synthetic_qa.json              # ReAct 自评改写模式
    python3 eval/run_eval.py --compare --load synthetic_qa.json                 # 三种模式对比
    python3 eval/run_eval.py --compare --load synthetic_qa.json --skip-ragas    # 只跑 IR 指标
    python3 eval/run_eval.py --mode full --difficulty multi-hop --n 10 --skip-ir  # 只跑 RAGAS, 指定难度和条数

输出:
    eval/data/report_<timestamp>_<mode>.json  — 评测报告
"""

# ── 兼容性补丁: 修复 ragas 与 langchain_community 的 vertexai 导入缺失 ──
import sys as _sys

_MODULE_NAME = "langchain_community.chat_models.vertexai"
if _MODULE_NAME not in _sys.modules:
    import types as _types
    _fake_module = _types.ModuleType(_MODULE_NAME)
    _fake_module.ChatVertexAI = type("ChatVertexAI", (), {})
    _sys.modules[_MODULE_NAME] = _fake_module

import os

# 解决 macOS 上多个库各自链接 OpenMP 的冲突
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 加载 .env
from pathlib import Path
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

from eval.dataset import build_eval_dataset
from eval.metrics import run_full_eval, run_react_eval, make_retrieval_func

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def print_report(report: dict, mode: str = ""):
    """打印评测报告到终端"""
    header = f"RAG Evaluation Report ({mode})" if mode else "RAG Evaluation Report"
    print("\n" + "=" * 60)
    print(header)
    print("=" * 60)
    print(f"Total queries: {report.get('total_queries', 0)}")

    # IR 指标 - overall
    ir = report.get("ir_metrics", {})
    if ir:
        print("\n── IR Metrics (overall) ──")
        for name in sorted(ir):
            print(f"  {name:20s}: {ir[name]:.4f}")

    # IR 按 difficulty 分层
    ir_by_diff = report.get("ir_by_difficulty", {})
    if ir_by_diff:
        print("\n── IR Metrics (by difficulty) ──")
        difficulties = sorted(ir_by_diff.keys())
        # 表头
        metrics_to_show = ["precision@5", "recall@5", "f1@5", "mrr"]
        header_parts = [f"{'':20s}"]
        for diff in difficulties:
            count = ir_by_diff[diff].get("count", 0)
            header_parts.append(f"{diff}(n={count}):>16s")
        print("  " + "".join(header_parts))

        for metric in metrics_to_show:
            parts = [f"  {metric:20s}"]
            for diff in difficulties:
                val = ir_by_diff[diff].get(metric, 0.0)
                parts.append(f"{val:>16.4f}")
            print("".join(parts))

    # RAGAS 指标
    ragas = report.get("ragas_metrics", {})
    if ragas:
        print("\n── RAGAS Metrics ──")
        for name in sorted(ragas):
            print(f"  {name:20s}: {ragas[name]:.4f}")

    # ReAct 迭代统计
    react_stats = report.get("react_stats", {})
    if react_stats:
        print("\n── ReAct Stats ──")
        print(f"  avg_iterations:     {react_stats.get('avg_iterations', 0):.1f}")
        print(f"  max_iterations:     {react_stats.get('max_iterations', 0)}")
        print(f"  multi_iter_pct:     {react_stats.get('multi_iteration_pct', 0):.1f}%")

    elapsed = report.get("elapsed_seconds", 0)
    if elapsed:
        print(f"\nElapsed: {elapsed:.1f}s")
    print("=" * 60)


def print_comparison_n(reports: dict):
    """打印 N 模式对比报告（支持 baseline/full/react）"""
    modes = list(reports.keys())
    first = reports[modes[0]]

    print("\n" + "=" * 80)
    print(f"COMPARISON: {' vs '.join(modes)}")
    print("=" * 80)
    print(f"Dataset: {first.get('total_queries', 0)} queries (same dataset)")

    def delta_str(base_val, val):
        d = val - base_val
        arrow = "↑" if d > 0.001 else ("↓" if d < -0.001 else "→")
        sign = "+" if d > 0 else ""
        return f"{sign}{d:.4f}{arrow}"

    # IR 对比
    ir_data = {m: reports[m].get("ir_metrics", {}) for m in modes}
    if any(ir_data.values()):
        print(f"\n── IR Metrics ──")
        header = f"  {'':20s}" + "".join(f"{m:>12s}" for m in modes)
        if len(modes) > 1:
            header += f"{'  delta(last-first)':>20s}"
        print(header)
        all_keys = sorted(set().union(*(d.keys() for d in ir_data.values())))
        for name in all_keys:
            vals = [ir_data[m].get(name, 0.0) for m in modes]
            line = f"  {name:20s}" + "".join(f"{v:>12.4f}" for v in vals)
            if len(vals) > 1:
                line += f"{delta_str(vals[0], vals[-1]):>20s}"
            print(line)

    # RAGAS 对比
    ragas_data = {m: reports[m].get("ragas_metrics", {}) for m in modes}
    if any(ragas_data.values()):
        print(f"\n── RAGAS Metrics ──")
        header = f"  {'':20s}" + "".join(f"{m:>12s}" for m in modes)
        if len(modes) > 1:
            header += f"{'  delta(last-first)':>20s}"
        print(header)
        all_keys = sorted(set().union(*(d.keys() for d in ragas_data.values())))
        for name in all_keys:
            vals = [ragas_data[m].get(name, 0.0) for m in modes]
            line = f"  {name:20s}" + "".join(f"{v:>12.4f}" for v in vals)
            if len(vals) > 1:
                line += f"{delta_str(vals[0], vals[-1]):>20s}"
            print(line)

    # IR 按 difficulty 对比
    diff_data = {m: reports[m].get("ir_by_difficulty", {}) for m in modes}
    if any(diff_data.values()):
        all_diffs = sorted(set().union(*(d.keys() for d in diff_data.values())))
        metrics_to_show = ["precision@5", "recall@5", "mrr"]
        for metric in metrics_to_show:
            print(f"\n── {metric} by difficulty ──")
            header = f"  {'difficulty':20s}" + "".join(f"{m:>12s}" for m in modes)
            if len(modes) > 1:
                header += f"{'  delta(last-first)':>20s}"
            print(header)
            for diff in all_diffs:
                vals = [diff_data[m].get(diff, {}).get(metric, 0.0) for m in modes]
                count = diff_data[modes[0]].get(diff, {}).get("count", 0)
                line = f"  {diff + f'(n={count})':20s}" + "".join(f"{v:>12.4f}" for v in vals)
                if len(vals) > 1:
                    line += f"{delta_str(vals[0], vals[-1]):>20s}"
                print(line)

    # ReAct 迭代统计
    react_stats = reports.get("react", {}).get("react_stats", {})
    if react_stats:
        print(f"\n── ReAct Iteration Stats ──")
        print(f"  avg_iterations:     {react_stats.get('avg_iterations', 0):.1f}")
        print(f"  max_iterations:     {react_stats.get('max_iterations', 0)}")
        print(f"  multi_iter_pct:     {react_stats.get('multi_iteration_pct', 0):.1f}%")

    print("=" * 80)


def run_single(dataset, mode: str, skip_ragas: bool = False, skip_ir: bool = False, n: int = 50) -> dict:
    """跑单次评测"""
    start = time.time()
    if mode == "react":
        report = run_react_eval(dataset, skip_ragas=skip_ragas, skip_ir=skip_ir)
    else:
        retrieval_func = make_retrieval_func(mode)
        report = run_full_eval(dataset, skip_ragas=skip_ragas, skip_ir=skip_ir, retrieval_func=retrieval_func)
    report["mode"] = mode
    report["elapsed_seconds"] = round(time.time() - start, 1)
    report["config"] = {"n_synthetic": n, "seed": 42}
    return report


def save_report(report: dict, suffix: str = "") -> str:
    """保存报告到文件"""
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    name = f"report_{ts}_{suffix}.json" if suffix else f"report_{ts}.json"
    output_path = os.path.join(DATA_DIR, name)
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"Report saved to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="RAG Evaluation Runner")
    parser.add_argument("--n", type=int, default=50, help="合成数据条数 (default: 50)")
    parser.add_argument("--load", type=str, default=None, help="从已有 JSON 文件加载")
    parser.add_argument("--skip-ragas", action="store_true", help="跳过 RAGAS 指标")
    parser.add_argument("--skip-ir", action="store_true", help="跳过 IR 指标（只跑 RAGAS）")
    parser.add_argument("--mode", type=str, default="full", choices=["full", "baseline", "react"],
                        help="实验模式: full=全优化, baseline=无优化, react=ReAct自评改写")
    parser.add_argument("--compare", action="store_true",
                        help="对比模式: 同一数据集跑 baseline/full/react 三种模式")
    parser.add_argument("--difficulty", type=str, default=None, choices=["simple", "complex", "multi-hop"],
                        help="只评测指定难度的数据")
    parser.add_argument("--output", type=str, default=None, help="报告输出路径")
    args = parser.parse_args()

    # 1. 加载/生成数据集（只生成一次，保证 baseline 和 full 用同一数据集）
    if args.load:
        path = args.load if os.path.isabs(args.load) else os.path.join(DATA_DIR, args.load)
        if os.path.exists(path):
            with open(path) as f:
                dataset = json.load(f)
            logger.info(f"Loaded {len(dataset)} QA pairs from {path}")
        else:
            logger.error(f"File not found: {path}")
            sys.exit(1)
    else:
        # 不指定 --difficulty 时，--n 直接限制生成数量
        # 指定 --difficulty 时，多生成一些以保证过滤后够用
        gen_n = args.n * 4 if args.difficulty else args.n
        dataset = build_eval_dataset(n=gen_n, seed=42)

    if not dataset:
        logger.error("Empty dataset. Make sure the knowledge base has documents.")
        sys.exit(1)

    # 按难度过滤（在 --n 限制之前）
    if args.difficulty:
        dataset = [d for d in dataset if d.get("difficulty") == args.difficulty]
        if not dataset:
            logger.error(f"No data with difficulty '{args.difficulty}'")
            sys.exit(1)
        logger.info(f"Filtered to {len(dataset)} queries with difficulty='{args.difficulty}'")

    # --n 限制条数（在过滤之后生效）
    if args.n and len(dataset) > args.n:
        random.seed(42)
        random.shuffle(dataset)
        dataset = dataset[:args.n]
        logger.info(f"Sampled {args.n} queries from filtered dataset")

    if args.compare:
        # 对比模式：同一数据集跑 baseline / full / react
        modes = ["baseline", "full", "react"]
        logger.info(f"=== Compare mode: running {' / '.join(modes)} on {len(dataset)} queries ===")

        reports = {}
        for mode in modes:
            logger.info(f"Running {mode}...")
            reports[mode] = run_single(dataset, mode, args.skip_ragas, args.skip_ir, args.n)
            save_report(reports[mode], mode)

        for mode in modes:
            print_report(reports[mode], mode)

        print_comparison_n(reports)

        for mode in modes:
            print(f"\n{mode} report: {os.path.join(DATA_DIR, f'report_')}*_{mode}.json")
    else:
        # 单次评测
        report = run_single(dataset, args.mode, args.skip_ragas, args.skip_ir, args.n)
        output_path = args.output or save_report(report, args.mode)
        if not args.output:
            pass  # already saved
        else:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

        print_report(report, args.mode)
        print(f"\nFull report: {output_path}")


if __name__ == "__main__":
    main()
