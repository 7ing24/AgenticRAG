#!/usr/bin/env python
"""评测集生成脚本

用法:
    python eval/build_dataset.py                    # 默认生成 100 条
    python eval/build_dataset.py --n 50             # 生成 50 条
    python eval/build_dataset.py --seed 123         # 指定种子
    python eval/build_dataset.py --output my_qa.json # 指定输出文件名
    python eval/build_dataset.py --preview           # 预览模式，不保存

输出:
    eval/data/synthetic_qa.json  — 评测数据集
"""

import sys as _sys

_MODULE_NAME = "langchain_community.chat_models.vertexai"
if _MODULE_NAME not in _sys.modules:
    import types as _types
    _fake_module = _types.ModuleType(_MODULE_NAME)
    _fake_module.ChatVertexAI = type("ChatVertexAI", (), {})
    _sys.modules[_MODULE_NAME] = _fake_module

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

from eval.dataset import EvalDatasetBuilder, DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def print_stats(dataset: list):
    """打印数据集统计"""
    from collections import Counter
    difficulties = Counter(d["difficulty"] for d in dataset)

    print(f"\n{'=' * 60}")
    print(f"评测集统计")
    print(f"{'=' * 60}")
    print(f"  总条数:     {len(dataset)}")
    print(f"  simple:     {difficulties.get('simple', 0)}")
    print(f"  complex:    {difficulties.get('complex', 0)}")
    print(f"  multi-hop:  {difficulties.get('multi-hop', 0)}")

    # 统计 relevant_chunk_ids 数量分布
    id_counts = [len(d.get("relevant_chunk_ids", [])) for d in dataset]
    if id_counts:
        avg_ids = sum(id_counts) / len(id_counts)
        max_ids = max(id_counts)
        print(f"  平均相关chunk数: {avg_ids:.1f}")
        print(f"  最大相关chunk数: {max_ids}")

    # 统计 ground_truth 长度
    gt_lens = [len(d.get("ground_truth", "")) for d in dataset]
    if gt_lens:
        avg_gt = sum(gt_lens) / len(gt_lens)
        print(f"  ground_truth平均长度: {avg_gt:.0f} 字")

    # 统计 evidence 条数
    ev_counts = [len(d.get("evidence", {})) for d in dataset]
    if ev_counts:
        avg_ev = sum(ev_counts) / len(ev_counts)
        print(f"  evidence平均条数: {avg_ev:.1f}")

    print(f"{'=' * 60}")


def print_samples(dataset: list, n: int = 3):
    """打印几条样本"""
    print(f"\n{'=' * 60}")
    print(f"样本预览 (前 {n} 条)")
    print(f"{'=' * 60}")
    for item in dataset[:n]:
        print(f"\n[{item['difficulty']}] {item['question']}")
        print(f"  relevant_chunk_ids: {item['relevant_chunk_ids']}")
        print(f"  ground_truth: {item['ground_truth'][:100]}...")
        for cid, ev in item.get("evidence", {}).items():
            print(f"  evidence[{cid}]: {ev[:80]}...")
    print()


def main():
    parser = argparse.ArgumentParser(description="评测集生成脚本")
    parser.add_argument("--n", type=int, default=100, help="生成条数 (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (default: 42)")
    parser.add_argument("--output", type=str, default="synthetic_qa.json", help="输出文件名")
    parser.add_argument("--preview", action="store_true", help="预览模式，不保存到文件")
    parser.add_argument("--no-cache", action="store_true", help="忽略已有缓存，强制重新生成")
    args = parser.parse_args()

    # 检查 API Key
    from core.config import config
    if not config.DASHSCOPE_API_KEY:
        logger.error("DASHSCOPE_API_KEY not set in .env")
        sys.exit(1)

    # 检查是否已有缓存
    if not args.no_cache:
        builder = EvalDatasetBuilder(seed=args.seed)
        existing = builder.load(args.output)
        if len(existing) >= args.n and builder._validate(existing[:args.n]):
            print(f"\n已存在缓存 ({len(existing)} 条)，使用缓存。加 --no-cache 强制重新生成。")
            dataset = existing[:args.n]
            print_stats(dataset)
            print_samples(dataset)
            return

    # 生成
    print(f"\n开始生成评测集: n={args.n}, seed={args.seed}")
    start_time = datetime.now()

    builder = EvalDatasetBuilder(seed=args.seed)
    dataset = builder.build(n=args.n, output_name=args.output)

    elapsed = (datetime.now() - start_time).total_seconds()

    if not dataset:
        logger.error("生成失败，请检查知识库是否有文档")
        sys.exit(1)

    print_stats(dataset)
    print_samples(dataset)

    if args.preview:
        print(f"预览模式，未保存。耗时 {elapsed:.1f}s")
    else:
        output_path = os.path.join(DATA_DIR, args.output)
        print(f"已保存到: {output_path}")
        print(f"耗时: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
