#!/usr/bin/env python
"""RAG 评测入口脚本

用法:
    python eval/run_eval.py                          # 默认 50 条合成数据
    python eval/run_eval.py --n 100                   # 100 条合成数据
    python eval/run_eval.py --load synthetic_qa.json  # 从已有数据集加载

输出:
    eval/data/report_<timestamp>.json   — 完整评测报告
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
import sys
import time
from datetime import datetime

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 加载 .env，确保独立运行时也能读取环境变量
from pathlib import Path
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

from eval.dataset import build_eval_dataset
from eval.metrics import run_full_eval

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def print_report(report: dict):
    """打印评测报告到终端"""
    print("\n" + "=" * 60)
    print("RAG Evaluation Report")
    print("=" * 60)
    print(f"Total queries: {report.get('total_queries', 0)}")

    ir = report.get("ir_metrics", {})
    if ir:
        print("\n── IR Metrics ──")
        for name in sorted(ir):
            print(f"  {name:20s}: {ir[name]:.4f}")
        skipped = report.get("ir_skipped", 0)
        if skipped > 0:
            print(f"  ({skipped} queries skipped — no relevant_chunk_id)")

    ragas = report.get("ragas_metrics", {})
    if ragas:
        print("\n── RAGAS Metrics ──")
        for name in sorted(ragas):
            print(f"  {name:20s}: {ragas[name]:.4f}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="RAG Evaluation Runner")
    parser.add_argument("--n", type=int, default=50, help="合成数据条数 (default: 50)")
    parser.add_argument("--load", type=str, default=None, help="从已有 JSON 文件加载 (跳过数据集生成)")
    parser.add_argument("--skip-ragas", action="store_true", help="跳过 RAGAS 指标 (仅计算 IR)")
    parser.add_argument("--output", type=str, default=None, help="报告输出路径")
    args = parser.parse_args()

    # 1. 加载/生成数据集
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
        dataset = build_eval_dataset(n=args.n)

    if not dataset:
        logger.error("Empty dataset. Make sure the knowledge base has documents.")
        sys.exit(1)

    # 2. 跑评测
    start = time.time()
    report = run_full_eval(dataset, skip_ragas=args.skip_ragas)

    elapsed = time.time() - start
    report["elapsed_seconds"] = round(elapsed, 1)
    report["config"] = {"n_synthetic": args.n}

    # 3. 保存报告
    output_path = args.output or os.path.join(DATA_DIR, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"Report saved to {output_path}")

    # 4. 打印到终端
    print_report(report)
    print(f"\nFull report: {output_path} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
