"""临时复现脚本：用真实 ReActParser + 复刻 run_stream 循环逻辑，看是否重复输出 Final Answer

用法: cd python-service && python3 tests/repro_stream_loop.py
"""
import sys, os, json, importlib.util
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 直接按文件加载 react_parser.py，绕过 agent/__init__.py 的重依赖导入链
_spec = importlib.util.spec_from_file_location(
    "react_parser_standalone",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent", "react_parser.py"),
)
_parser_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_parser_mod)
ReActParser = _parser_mod.ReActParser

parser = ReActParser()

def stream_tokens(llm_response):
    """复刻 run_stream 里的流式状态机：检测到 'final answer:' 前静默缓冲"""
    streaming = False
    last_yielded_pos = 0
    out = []
    for chunk in llm_response:
        # chunk 每次一个字符
        last_yielded_pos = 0
        llm = ""
        break
    # 直接完整模拟（逐 token 累积，与真实 chunk 行为一致）
    llm = ""
    streaming = False
    last_yielded_pos = 0
    tokens = []
    for ch in llm_response:
        llm += ch
        if not streaming:
            keyword_idx = llm.lower().find("final answer:")
            if keyword_idx >= 0:
                streaming = True
                last_yielded_pos = keyword_idx + len("final answer:")
                while last_yielded_pos < len(llm) and llm[last_yielded_pos] in (' ', '\n', '\r'):
                    last_yielded_pos += 1
        if streaming and len(llm) > last_yielded_pos:
            tokens.append(llm[last_yielded_pos:])
            last_yielded_pos = len(llm)
    return "".join(tokens)

def simulate(responses, max_iterations=5):
    """复刻 run_stream 的迭代循环（只关注 token/end 事件与 break 逻辑）"""
    tokens_emitted = []
    final_answer = None
    iteration_llm_outputs = []
    for iteration in range(max_iterations):
        llm_response = responses[iteration] if iteration < len(responses) else 'Final Answer: 抱歉，无法继续。'
        iteration_llm_outputs.append(llm_response)
        # 流式 token 输出
        t = stream_tokens(llm_response)
        if t:
            tokens_emitted.append(t)
        # 解析
        parsed = parser.parse(llm_response)
        if parsed.is_final_answer:
            final_answer = parsed.final_answer
            break
        if parsed.action and parsed.tool_name:
            # 执行工具（模拟返回空）
            continue
        # else: 提示重试，继续循环
    if final_answer is None:
        final_answer = responses[-1] if len(responses) <= max_iterations else 'forced'
    return iteration_llm_outputs, tokens_emitted, final_answer

SCENARIOS = {
    "S1 正常: Action→Final Answer": [
        'Thought: 需要检索。\nAction: knowledge_search(query="公司")\n',
        'Thought: 已够。\nFinal Answer: 这是我们公司的介绍。\n',
    ],
    "S2 直接 Final Answer": [
        'Thought: 直接回答。\nFinal Answer: 答案在这里。\n',
    ],
    "S3 纯文本长答案(无标签)": [
        '这是完全没有任何 ReAct 标签的普通长文本答案，超过二十个字符，直接当作最终答案处理。',
    ],
    "S4 每次都是 Final Answer 但解析失败?": [
        'Final Answer: 第一次答案内容比较长，超过二十个字符。',
        'Final Answer: 第二次答案内容也比较长，超过二十个字符。',
    ],
    "S5 全角冒号 Final Answer：": [
        'Thought: 全角冒号场景。\nFinal Answer：这是全角冒号冒号后的答案内容，超过二十个字符。',
    ],
    "S6 简短无法解析 (抱歉)": [
        '抱歉',
        '抱歉',
        '抱歉',
        '抱歉',
        '抱歉',
    ],
}

for name, responses in SCENARIOS.items():
    outputs, tokens, final_answer = simulate(responses)
    token_text = "".join(tokens)
    print(f"\n=== {name} ===")
    print(f"  迭代次数: {len(outputs)}")
    print(f"  token拼接: {token_text!r}")
    print(f"  final_answer: {final_answer!r}")
    print(f"  token 中 'Final Answer' 出现次数: {token_text.lower().count('final answer')}")
