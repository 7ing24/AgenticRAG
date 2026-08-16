import os
import json
import requests
from typing import AsyncGenerator, Generator, Optional, Dict, Any
from langchain_community.llms import Tongyi
# from langchain_community.llms import Ollama  # 本地模型（已注释，改用 Tongyi）
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import BaseCallbackHandler
from PIL import Image
import pytesseract

# 使用统一配置管理模块
from core.config import config

# LLM Fallback 引擎
from core.llm_fallback import (
    fallback_handler, LLMServiceError, CircuitBreakerOpenError, TokenCallback as FallbackTokenCallback
)

# 配置Tesseract OCR路径（空值时由 parser.py 自动检测）
if config.TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_PATH

class _TokenUsage:
    """Token 用量辅助类 — 用于流式调用后记录 token 数"""
    def __init__(self, input_tokens, output_tokens):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = input_tokens + output_tokens

    def to_dict(self):
        return {
            "input_tokens": self.input_tokens or None,
            "output_tokens": self.output_tokens or None,
            "total_tokens": self.total_tokens or None,
        }


class LLMService:
    def __init__(self):
        self._last_token_callback = None
        # ============================================================
        # 阿里云通义千问（默认，需设置 DASHSCOPE_API_KEY）
        # ============================================================
        api_key = config.DASHSCOPE_API_KEY
        if not api_key:
            config.logger.warning("DASHSCOPE_API_KEY not found. LLM features will not work properly.")
            self.llm = None
        else:
            self.llm = Tongyi(
                model_name="qwen-plus",
                api_key=api_key,
                streaming=True
            )

        # ============================================================
        # Ollama 本地模型（免费，已注释）
        # ============================================================
        # ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        # ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        # self.llm = Ollama(
        #     model=ollama_model,
        #     base_url=ollama_base_url,
        # )
        # config.logger.info(f"LLM initialized: Ollama ({ollama_model}) at {ollama_base_url}")

        # 优化后的 Prompt 模板
        # 支持对话上下文和知识库上下文
        self.prompt = PromptTemplate.from_template(
            """
            你是一个专业的AI知识库助手。请根据提供的知识库信息回答用户问题。

            重要规则：
            1. 如果知识库中有相关信息，请优先引用知识库内容进行回答
            2. 如果知识库中没有相关信息，请先说明"知识库中未找到相关信息，以下是我的理解："，然后根据你的知识进行回答
            3. 如果是问候、自我介绍等问题，可以直接回答，不需要强行引用知识库
            4. 回答要自然、友好，避免机械和死板
            5. 不要提及"AI服务不可用"、"系统错误"等技术问题，你始终处于正常工作状态
            6. 禁止在回答中出现"doc_id"、内部ID等系统技术信息。引用来源时用文档名称即可
            7. 使用简洁的排版，段落之间最多空一行，不要堆砌大量空行

            对话历史（仅供参考，可能包含过时信息）：
            {conversation_context}

            相关知识库：
            {knowledge_context}

            用户当前问题：
            {question}

            请给出自然、友好的回答：
            """
        )

        # 标题生成模板
        self.summary_prompt = PromptTemplate.from_template(
            """
            请为以下用户问题生成一个简短的标题（Summary）。
            
            用户问题：
            {question}
            
            要求：
            1. 标题应概括问题的主要内容。
            2. 长度控制在10个字以内。
            3. 不需要任何前缀或后缀，直接返回标题文本。
            
            标题：
            """
        )

    """
     * 获取 LLM 的回答
     * @param question 用户问题
     * @param context_docs 上下文文档列表
     * @param conversation_context 对话上下文（可选）
     * @return LLM 的回答
     * """
    def get_answer(self, question: str, context_docs: list, conversation_context: str = "") -> str:
        import time
        start_time = time.time()
        
        if not self.llm:
            config.logger.info(f"LLM get_answer completed in {time.time() - start_time:.4f}s (no API key)")
            return fallback_handler.registry.get("no_api_key")

        # 处理包含图片的问题
        image_process_start = time.time()
        processed_question = self.process_question_with_images(question)
        image_process_time = time.time() - image_process_start
        config.logger.info(f"Image processing completed in {image_process_time:.4f}s")

        # 处理知识库上下文
        if not context_docs:
            knowledge_context = "（无相关知识库信息）"
        else:
            knowledge_context = "\n\n".join([
                doc.page_content if hasattr(doc, 'page_content') else str(doc)
                for doc in context_docs
            ])

        # 处理对话上下文 - 兼容 dict 和 str
        if isinstance(conversation_context, dict):
            conversation_context = conversation_context.get("text", "")

        # 过滤掉错误信息
        cleaned_context = self.clean_conversation_context(conversation_context)
        if not cleaned_context or cleaned_context.strip() == "":
            cleaned_context = "（无对话历史）"

        # 用模板格式化 prompt
        prompt_text = self.prompt.format(
            conversation_context=cleaned_context,
            knowledge_context=knowledge_context,
            question=processed_question
        )

        try:
            llm_start = time.time()
            result, token_cb = self._dashscope_generate(prompt_text)
            self._last_token_callback = token_cb
            config.logger.info(
                f"[TokenUsage][answer] input:{token_cb.input_tokens} "
                f"output:{token_cb.output_tokens} total:{token_cb.total_tokens}"
            )
            llm_time = time.time() - llm_start
            config.logger.info(f"LLM invocation completed in {llm_time:.4f}s")
            config.logger.info(f"LLM get_answer completed in {time.time() - start_time:.4f}s")
            return result
        except Exception as e:
            config.logger.error(f"LLM Error: {e}")
            config.logger.info(f"LLM get_answer completed in {time.time() - start_time:.4f}s (error)")
            return fallback_handler.registry.get("generation")

    def clean_conversation_context(self, context: str) -> str:
        """
        清理对话上下文，移除错误信息，防止污染后续回答
        """
        if not context:
            return ""
        
        # 需要过滤的错误关键词
        error_keywords = [
            "AI服务暂时不可用",
            "服务不可用",
            "系统错误",
            "无法连接",
            "网络错误",
            "超时",
            "API密钥",
            "配置错误"
        ]
        
        # 按行分割
        lines = context.split("\n")
        # 过滤包含错误关键词的行
        cleaned_lines = [
            line for line in lines 
            if not any(keyword in line for keyword in error_keywords)
        ]
        
        return "\n".join(cleaned_lines)

    """
     * 流式获取 LLM 的回答
     * @param question 用户问题
     * @param context_docs 上下文文档列表
     * @param conversation_context 对话上下文（可选）
     * @return 流式生成器，逐个token返回
     * """
    def get_answer_stream(self, question: str, context_docs: list, conversation_context: str = "") -> Generator[str, None, None]:
        import time
        start_time = time.time()
        
        if not self.llm:
            config.logger.info(f"LLM get_answer_stream completed in {time.time() - start_time:.4f}s (no API key)")
            yield json.dumps({"type": "error", "content": fallback_handler.registry.get("no_api_key")})
            return

        # 处理包含图片的问题
        image_process_start = time.time()
        processed_question = self.process_question_with_images(question)
        image_process_time = time.time() - image_process_start
        config.logger.info(f"Image processing completed in {image_process_time:.4f}s")

        # 处理知识库上下文
        if not context_docs:
            knowledge_context = "（无相关知识库信息）"
        else:
            knowledge_context = "\n\n".join([
                doc.page_content if hasattr(doc, 'page_content') else str(doc)
                for doc in context_docs
            ])

        # 处理对话上下文 - 兼容 dict 和 str
        if isinstance(conversation_context, dict):
            conversation_context = conversation_context.get("text", "")

        # 过滤掉错误信息
        cleaned_context = self.clean_conversation_context(conversation_context)
        if not cleaned_context or cleaned_context.strip() == "":
            cleaned_context = "（无对话历史）"

        # 构建处理链
        chain = (
            self.prompt
            | self.llm
            | StrOutputParser()
        )

        # 计算 prompt 文本用于 token 计数
        prompt_text = self.prompt.format(
            conversation_context=cleaned_context,
            knowledge_context=knowledge_context,
            question=processed_question
        )
        input_tokens = self.llm.get_num_tokens(prompt_text) if self.llm else 0

        try:
            # 发送开始信号
            yield json.dumps({"type": "start", "content": ""})

            # 流式调用
            llm_start = time.time()
            full_response = ""
            for chunk in chain.stream({
                "conversation_context": cleaned_context,
                "knowledge_context": knowledge_context,
                "question": processed_question
            }):
                full_response += chunk
                yield json.dumps({"type": "token", "content": chunk})
            llm_time = time.time() - llm_start
            config.logger.info(f"LLM stream invocation completed in {llm_time:.4f}s")

            # 用 get_num_tokens 计算 token 用量（比 callback 可靠）
            output_tokens = self.llm.get_num_tokens(full_response) if self.llm else 0
            self._last_token_callback = _TokenUsage(input_tokens, output_tokens)

            # 发送结束信号
            yield json.dumps({"type": "end", "content": full_response})
            config.logger.info(f"LLM get_answer_stream completed in {time.time() - start_time:.4f}s")

        except Exception as e:
            config.logger.error(f"LLM Stream Error: {e}")
            config.logger.info(f"LLM get_answer_stream completed in {time.time() - start_time:.4f}s (error)")
            yield json.dumps({"type": "error", "content": fallback_handler.registry.get("generation")})

    def generate_title(self, question: str) -> str:
        import time
        start_time = time.time()
        
        if not self.llm:
            config.logger.info(f"LLM generate_title completed in {time.time() - start_time:.4f}s (no API key)")
            return fallback_handler.registry.get("title")

        chain = (
            self.summary_prompt
            | self.llm
            | StrOutputParser()
        )
        
        try:
            llm_start = time.time()
            title = chain.invoke({"question": question})
            llm_time = time.time() - llm_start
            # 清理可能的额外空白或引号
            result = title.strip().strip('"').strip("'")
            config.logger.info(f"LLM title generation completed in {llm_time:.4f}s")
            config.logger.info(f"LLM generate_title completed in {time.time() - start_time:.4f}s")
            return result
        except Exception as e:
            config.logger.error(f"LLM Title Generation Error: {e}")
            config.logger.info(f"LLM generate_title completed in {time.time() - start_time:.4f}s (error)")
            return fallback_handler.registry.get("title")

    def extract_text_from_image(self, image_url: str) -> str:
        """
        从图片URL中提取文字
        """
        try:
            # 处理相对路径，转换为完整URL
            if image_url.startswith('/api/'):
                # 使用后端服务地址
                image_url = f"http://localhost:8080{image_url}"
            
            config.logger.info(f"Downloading image from: {image_url}")
            
            # 下载图片
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            
            # 保存到临时文件
            temp_path = os.path.join(config.TEMP_DIR, "temp_image.png")
            with open(temp_path, "wb") as f:
                f.write(response.content)
            
            config.logger.info(f"Image saved to temp file, size: {len(response.content)} bytes")
            
            # 使用OCR提取文字
            image = Image.open(temp_path)
            text = pytesseract.image_to_string(image, lang='chi_sim+eng')
            
            config.logger.info(f"OCR result: {text[:100]}...")  # 打印前100个字符
            
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return text.strip() if text.strip() else "图片中未识别到文字"
        except Exception as e:
            config.logger.error(f"Error extracting text from image: {e}")
            return f"无法从图片中提取文字: {str(e)}"

    def process_question_with_images(self, question: str) -> str:
        """
        处理包含图片URL的问题，提取图片中的文字并添加到问题中
        """
        import re
        # 查找图片URL（支持完整URL和相对路径）
        image_urls = re.findall(r'图片URL: (/api/[^\n]+)', question)
        
        config.logger.info(f"Found image URLs: {image_urls}")
        
        if image_urls:
            processed_question = question
            for image_url in image_urls:
                # 提取图片中的文字
                image_text = self.extract_text_from_image(image_url)
                # 将图片文字添加到问题中
                processed_question += f"\n\n图片内容: {image_text}"
            return processed_question
        else:
            return question

    def _dashscope_generate(self, prompt_text: str):
        """直接调 DashScope API，带重试 + 熔断保护

        Returns: (text, TokenCallback)
        """
        import dashscope

        api_key = config.DASHSCOPE_API_KEY
        model = getattr(config, 'DASHSCOPE_MODEL', 'qwen-plus')

        if not api_key:
            return fallback_handler.registry.get("no_api_key"), FallbackTokenCallback()

        def _call():
            resp = dashscope.Generation.call(
                model=model,
                prompt=prompt_text,
                api_key=api_key,
                result_format='message',
            )
            cb = TokenCallback()
            if resp.status_code == 200 and resp.output and resp.output.choices:
                text = resp.output.choices[0].message.content
                if resp.usage:
                    cb.input_tokens = resp.usage.input_tokens
                    cb.output_tokens = resp.usage.output_tokens
                    cb.total_tokens = resp.usage.total_tokens
                return text, cb
            else:
                code = getattr(resp, 'status_code', 'unknown')
                msg = getattr(resp, 'message', '')
                raise LLMServiceError(f"DashScope API error: status={code} message={msg}")

        try:
            return fallback_handler.invoke(_call, scene="generation")
        except CircuitBreakerOpenError:
            return fallback_handler.registry.get("circuit_open"), FallbackTokenCallback()
        except LLMServiceError:
            return fallback_handler.registry.get("generation"), FallbackTokenCallback()

    def get_last_token_usage(self) -> Dict[str, Optional[int]]:
        """返回最近一次 LLM 调用的 token 用量"""
        if self._last_token_callback:
            return self._last_token_callback.to_dict()
        return {}

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 150) -> str:
        """
        简单的文本生成方法（用于闲聊等场景）
        """
        import time
        start_time = time.time()
        
        if not self.llm:
            config.logger.info(f"LLM generate completed in {time.time() - start_time:.4f}s (no API key)")
            return fallback_handler.registry.get("no_api_key")
        
        try:
            # 使用简单的 prompt
            result, token_cb = self._dashscope_generate(prompt)
            self._last_token_callback = token_cb

            config.logger.info(f"LLM generate completed in {time.time() - start_time:.4f}s")
            return result
        except Exception as e:
            config.logger.error(f"LLM generate Error: {e}")
            return fallback_handler.registry.get("generation")


class TokenCallback(BaseCallbackHandler):
    """捕获 LLM 调用的 token 用量"""
    def __init__(self):
        self.input_tokens = None
        self.output_tokens = None
        self.total_tokens = None

    def on_llm_end(self, response, **kwargs):
        # 尝试从 llm_output 中提取
        if hasattr(response, 'llm_output') and response.llm_output:
            usage = response.llm_output.get('token_usage', {})
            self.input_tokens = usage.get('prompt_tokens') or usage.get('input_tokens')
            self.output_tokens = usage.get('completion_tokens') or usage.get('output_tokens')
            self.total_tokens = usage.get('total_tokens')
        # 也尝试从 generation_info 中提取
        for gen_list in response.generations:
            for gen in gen_list:
                info = gen.generation_info or {}
                um = info.get('usage_metadata', {})
                if um:
                    self.input_tokens = self.input_tokens or um.get('input_tokens')
                    self.output_tokens = self.output_tokens or um.get('output_tokens')
                    self.total_tokens = self.total_tokens or um.get('total_tokens')
        import logging
        logging.getLogger(__name__).info(
            f"[TokenCallback] tokens captured — input:{self.input_tokens} output:{self.output_tokens} total:{self.total_tokens}"
        )

    def to_dict(self) -> Dict[str, Optional[int]]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }

# 创建单例实例
llm_service = LLMService()

# 导出（保持兼容性）
llm = llm_service