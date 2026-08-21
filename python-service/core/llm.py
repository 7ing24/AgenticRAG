import os
import json
import threading
import requests
from typing import AsyncGenerator, Generator, Optional, Dict, Any, List
from langchain_community.llms import Tongyi
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import BaseCallbackHandler
from PIL import Image
import pytesseract

from core.config import config
from core.llm_fallback import (
    fallback_handler,
    LLMServiceError,
    CircuitBreakerOpenError,
    TokenCallback as FallbackTokenCallback,
)

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
    MAX_OUTPUT_TOKENS = 2048
    REACT_STOP = ["\nThought:", "\nObservation:"]

    def __init__(self):
        self._local = threading.local()
        api_key = config.DASHSCOPE_API_KEY
        if not api_key:
            config.logger.warning(
                "DASHSCOPE_API_KEY not found. LLM features will not work properly."
            )
            self.llm = None
        else:
            self.llm = Tongyi(
                model_name=config.DASHSCOPE_MODEL, api_key=api_key, streaming=True
            )

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

    def get_answer(
        self,
        question: str,
        context_docs: list,
        conversation_context: str = "",
    ) -> str:
        import time

        start_time = time.time()

        if not self.llm:
            config.logger.info(
                f"LLM get_answer completed in {time.time() - start_time:.4f}s (no API key)"
            )
            return fallback_handler.registry.get("no_api_key")

        image_process_start = time.time()
        processed_question = self.process_question_with_images(question)
        image_process_time = time.time() - image_process_start
        config.logger.info(
            f"Image processing completed in {image_process_time:.4f}s"
        )

        if not context_docs:
            knowledge_context = "（无相关知识库信息）"
        else:
            knowledge_context = "\n\n".join(
                [
                    (
                        doc.page_content
                        if hasattr(doc, "page_content")
                        else str(doc)
                    )
                    for doc in context_docs
                ]
            )

        if isinstance(conversation_context, dict):
            conversation_context = conversation_context.get("text", "")

        cleaned_context = self.clean_conversation_context(conversation_context)
        if not cleaned_context or cleaned_context.strip() == "":
            cleaned_context = "（无对话历史）"

        prompt_text = self.prompt.format(
            conversation_context=cleaned_context,
            knowledge_context=knowledge_context,
            question=processed_question,
        )

        try:
            llm_start = time.time()
            result, token_cb = self._dashscope_generate(prompt_text)
            self.set_token_usage(token_cb)
            config.logger.info(
                f"[TokenUsage][answer] input:{token_cb.input_tokens} "
                f"output:{token_cb.output_tokens} total:{token_cb.total_tokens}"
            )
            llm_time = time.time() - llm_start
            config.logger.info(f"LLM invocation completed in {llm_time:.4f}s")
            config.logger.info(
                f"LLM get_answer completed in {time.time() - start_time:.4f}s"
            )
            return result
        except Exception as e:
            config.logger.error(f"LLM Error: {e}")
            config.logger.info(
                f"LLM get_answer completed in {time.time() - start_time:.4f}s (error)"
            )
            return fallback_handler.registry.get("generation")

    def clean_conversation_context(self, context: str) -> str:
        if not context:
            return ""

        error_keywords = [
            "AI服务暂时不可用",
            "服务不可用",
            "系统错误",
            "无法连接",
            "网络错误",
            "超时",
            "API密钥",
            "配置错误",
        ]

        lines = context.split("\n")
        cleaned_lines = [
            line
            for line in lines
            if not any(keyword in line for keyword in error_keywords)
        ]
        return "\n".join(cleaned_lines)

    def get_answer_stream(
        self,
        question: str,
        context_docs: list,
        conversation_context: str = "",
    ) -> Generator[str, None, None]:
        import time

        start_time = time.time()

        if not self.llm:
            config.logger.info(
                f"LLM get_answer_stream completed in {time.time() - start_time:.4f}s (no API key)"
            )
            yield json.dumps(
                {
                    "type": "error",
                    "content": fallback_handler.registry.get("no_api_key"),
                }
            )
            return

        image_process_start = time.time()
        processed_question = self.process_question_with_images(question)
        image_process_time = time.time() - image_process_start
        config.logger.info(
            f"Image processing completed in {image_process_time:.4f}s"
        )

        if not context_docs:
            knowledge_context = "（无相关知识库信息）"
        else:
            knowledge_context = "\n\n".join(
                [
                    (
                        doc.page_content
                        if hasattr(doc, "page_content")
                        else str(doc)
                    )
                    for doc in context_docs
                ]
            )

        if isinstance(conversation_context, dict):
            conversation_context = conversation_context.get("text", "")

        cleaned_context = self.clean_conversation_context(conversation_context)
        if not cleaned_context or cleaned_context.strip() == "":
            cleaned_context = "（无对话历史）"

        chain = self.prompt | self.llm | StrOutputParser()

        prompt_text = self.prompt.format(
            conversation_context=cleaned_context,
            knowledge_context=knowledge_context,
            question=processed_question,
        )
        input_tokens = self.llm.get_num_tokens(prompt_text) if self.llm else 0

        try:
            yield json.dumps({"type": "start", "content": ""})

            llm_start = time.time()
            full_response = ""
            for chunk in chain.stream(
                {
                    "conversation_context": cleaned_context,
                    "knowledge_context": knowledge_context,
                    "question": processed_question,
                }
            ):
                full_response += chunk
                yield json.dumps({"type": "token", "content": chunk})
            llm_time = time.time() - llm_start
            config.logger.info(
                f"LLM stream invocation completed in {llm_time:.4f}s"
            )

            output_tokens = (
                self.llm.get_num_tokens(full_response) if self.llm else 0
            )
            self.set_token_usage(_TokenUsage(input_tokens, output_tokens))

            yield json.dumps({"type": "end", "content": full_response})
            config.logger.info(
                f"LLM get_answer_stream completed in {time.time() - start_time:.4f}s"
            )

        except Exception as e:
            config.logger.error(f"LLM Stream Error: {e}")
            config.logger.info(
                f"LLM get_answer_stream completed in {time.time() - start_time:.4f}s (error)"
            )
            yield json.dumps(
                {
                    "type": "error",
                    "content": fallback_handler.registry.get("generation"),
                }
            )

    def generate_title(self, question: str) -> str:
        import time

        start_time = time.time()

        if not self.llm:
            config.logger.info(
                f"LLM generate_title completed in {time.time() - start_time:.4f}s (no API key)"
            )
            return fallback_handler.registry.get("title")

        chain = self.summary_prompt | self.llm | StrOutputParser()

        try:
            llm_start = time.time()
            title = chain.invoke({"question": question})
            llm_time = time.time() - llm_start
            result = title.strip().strip('"').strip("'")
            config.logger.info(
                f"LLM title generation completed in {llm_time:.4f}s"
            )
            config.logger.info(
                f"LLM generate_title completed in {time.time() - start_time:.4f}s"
            )
            return result
        except Exception as e:
            config.logger.error(f"LLM Title Generation Error: {e}")
            config.logger.info(
                f"LLM generate_title completed in {time.time() - start_time:.4f}s (error)"
            )
            return fallback_handler.registry.get("title")

    def extract_text_from_image(self, image_url: str) -> str:
        try:
            if image_url.startswith("/api/"):
                image_url = f"http://localhost:8080{image_url}"

            config.logger.info(f"Downloading image from: {image_url}")

            response = requests.get(image_url, timeout=10)
            response.raise_for_status()

            temp_path = os.path.join(config.TEMP_DIR, "temp_image.png")
            with open(temp_path, "wb") as f:
                f.write(response.content)

            config.logger.info(
                f"Image saved to temp file, size: {len(response.content)} bytes"
            )

            image = Image.open(temp_path)
            text = pytesseract.image_to_string(image, lang="chi_sim+eng")

            config.logger.info(f"OCR result: {text[:100]}...")

            if os.path.exists(temp_path):
                os.remove(temp_path)

            return text.strip() if text.strip() else "图片中未识别到文字"
        except Exception as e:
            config.logger.error(f"Error extracting text from image: {e}")
            return f"无法从图片中提取文字: {str(e)}"

    def process_question_with_images(self, question: str) -> str:
        import re

        image_urls = re.findall(r"图片URL: (/api/[^\n]+)", question)
        config.logger.info(f"Found image URLs: {image_urls}")

        if image_urls:
            processed_question = question
            for image_url in image_urls:
                image_text = self.extract_text_from_image(image_url)
                processed_question += f"\n\n图片内容: {image_text}"
            return processed_question
        else:
            return question

    def _dashscope_generate(self, prompt_text: str):
        """直接调 DashScope API，带重试 + 熔断保护"""
        import dashscope

        api_key = config.DASHSCOPE_API_KEY
        model = getattr(config, "DASHSCOPE_MODEL", "qwen-plus")

        if not api_key:
            return (
                fallback_handler.registry.get("no_api_key"),
                FallbackTokenCallback(),
            )

        def _call():
            resp = dashscope.Generation.call(
                model=model,
                prompt=prompt_text,
                api_key=api_key,
                result_format="message",
                max_tokens=self.MAX_OUTPUT_TOKENS,
                temperature=0.2,
                repetition_penalty=1.15,
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
                code = getattr(resp, "status_code", "unknown")
                msg = getattr(resp, "message", "")
                raise LLMServiceError(
                    f"DashScope API error: status={code} message={msg}"
                )

        try:
            return fallback_handler.invoke(_call, scene="generation")
        except CircuitBreakerOpenError:
            return (
                fallback_handler.registry.get("circuit_open"),
                FallbackTokenCallback(),
            )
        except LLMServiceError:
            return (
                fallback_handler.registry.get("generation"),
                FallbackTokenCallback(),
            )

    def chat_generate(self, messages: List[Dict[str, str]]) -> str:
        """按角色消息生成（支持 system 角色），返回文本并记录 token 用量。"""
        import time

        start_time = time.time()

        if not self.llm:
            config.logger.info(
                f"LLM chat_generate completed in {time.time() - start_time:.4f}s (no API key)"
            )
            return fallback_handler.registry.get("no_api_key")

        try:
            result, token_cb = self._dashscope_chat_generate(messages)
            self.set_token_usage(token_cb)
            config.logger.info(
                f"LLM chat_generate completed in {time.time() - start_time:.4f}s"
            )
            return result
        except Exception as e:
            config.logger.error(f"LLM chat_generate Error: {e}")
            return fallback_handler.registry.get("generation")

    def _dashscope_chat_generate(self, messages: List[Dict[str, str]]):
        """直接调 DashScope messages API，带重试 + 熔断保护"""
        import dashscope

        api_key = config.DASHSCOPE_API_KEY
        model = getattr(config, "DASHSCOPE_MODEL", "qwen-plus")

        if not api_key:
            return (
                fallback_handler.registry.get("no_api_key"),
                FallbackTokenCallback(),
            )

        def _call():
            resp = dashscope.Generation.call(
                model=model,
                messages=messages,
                api_key=api_key,
                result_format="message",
                max_tokens=self.MAX_OUTPUT_TOKENS,
                temperature=0.2,
                repetition_penalty=1.15,
                stop=self.REACT_STOP,
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
                code = getattr(resp, "status_code", "unknown")
                msg = getattr(resp, "message", "")
                raise LLMServiceError(
                    f"DashScope API error: status={code} message={msg}"
                )

        try:
            return fallback_handler.invoke(_call, scene="generation")
        except CircuitBreakerOpenError:
            return (
                fallback_handler.registry.get("circuit_open"),
                FallbackTokenCallback(),
            )
        except LLMServiceError:
            return (
                fallback_handler.registry.get("generation"),
                FallbackTokenCallback(),
            )

    def chat_stream(
        self, messages: List[Dict[str, str]]
    ) -> Generator[str, None, None]:
        """按角色消息流式生成，逐 token yield；结束前记录 token 用量。"""
        if not self.llm:
            yield fallback_handler.registry.get("no_api_key")
            return

        import dashscope

        api_key = config.DASHSCOPE_API_KEY
        model = getattr(config, "DASHSCOPE_MODEL", "qwen-plus")
        token_usage = None

        responses = dashscope.Generation.call(
            model=model,
            messages=messages,
            api_key=api_key,
            result_format="message",
            stream=True,
            incremental_output=True,
            max_tokens=self.MAX_OUTPUT_TOKENS,
            temperature=0.2,
            repetition_penalty=1.15,
            stop=self.REACT_STOP,
        )
        for resp in responses:
            if resp.status_code == 200 and resp.output and resp.output.choices:
                piece = resp.output.choices[0].message.content
                if resp.usage:
                    token_usage = resp.usage
                if piece:
                    yield piece
            else:
                code = getattr(resp, "status_code", "unknown")
                msg = getattr(resp, "message", "")
                raise LLMServiceError(
                    f"DashScope stream error: status={code} message={msg}"
                )

        if token_usage is not None:
            input_tokens = (
                getattr(token_usage, "input_tokens", 0)
                or getattr(token_usage, "prompt_tokens", 0)
                or 0
            )
            output_tokens = (
                getattr(token_usage, "output_tokens", 0)
                or getattr(token_usage, "completion_tokens", 0)
                or 0
            )
            self.set_token_usage(_TokenUsage(input_tokens, output_tokens))

    def generate_stream(
        self, prompt: str
    ) -> Generator[str, None, None]:
        """按单条 prompt 流式生成，逐 token yield；结束前记录 token 用量。

        与 chat_stream 的区别：不追加 REACT_STOP 截断（合成/摘要类场景需要完整输出）。
        """
        if not self.llm:
            yield fallback_handler.registry.get("no_api_key")
            return

        import dashscope

        api_key = config.DASHSCOPE_API_KEY
        model = getattr(config, "DASHSCOPE_MODEL", "qwen-plus")
        token_usage = None

        responses = dashscope.Generation.call(
            model=model,
            prompt=prompt,
            api_key=api_key,
            result_format="message",
            stream=True,
            incremental_output=True,
            max_tokens=self.MAX_OUTPUT_TOKENS,
            temperature=0.2,
            repetition_penalty=1.15,
        )
        for resp in responses:
            if resp.status_code == 200 and resp.output and resp.output.choices:
                piece = resp.output.choices[0].message.content
                if resp.usage:
                    token_usage = resp.usage
                if piece:
                    yield piece
            else:
                code = getattr(resp, "status_code", "unknown")
                msg = getattr(resp, "message", "")
                raise LLMServiceError(
                    f"DashScope stream error: status={code} message={msg}"
                )

        if token_usage is not None:
            input_tokens = (
                getattr(token_usage, "input_tokens", 0)
                or getattr(token_usage, "prompt_tokens", 0)
                or 0
            )
            output_tokens = (
                getattr(token_usage, "output_tokens", 0)
                or getattr(token_usage, "completion_tokens", 0)
                or 0
            )
            self.set_token_usage(_TokenUsage(input_tokens, output_tokens))

    def set_token_usage(self, token_cb):
        self._local.last_token_callback = token_cb

    def get_last_token_usage(self) -> Dict[str, Optional[int]]:
        token_cb = getattr(self._local, "last_token_callback", None)
        if token_cb:
            return token_cb.to_dict()
        return {}

    def generate(
        self, prompt: str, temperature: float = 0.7, max_tokens: int = 150
    ) -> str:
        import time

        start_time = time.time()

        if not self.llm:
            config.logger.info(
                f"LLM generate completed in {time.time() - start_time:.4f}s (no API key)"
            )
            return fallback_handler.registry.get("no_api_key")

        try:
            result, token_cb = self._dashscope_generate(prompt)
            self.set_token_usage(token_cb)
            config.logger.info(
                f"LLM generate completed in {time.time() - start_time:.4f}s"
            )
            return result
        except Exception as e:
            config.logger.error(f"LLM generate Error: {e}")
            return fallback_handler.registry.get("generation")


class TokenCallback(BaseCallbackHandler):

    def __init__(self):
        self.input_tokens = None
        self.output_tokens = None
        self.total_tokens = None

    def on_llm_end(self, response, **kwargs):
        if hasattr(response, "llm_output") and response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            self.input_tokens = usage.get("prompt_tokens") or usage.get(
                "input_tokens"
            )
            self.output_tokens = usage.get("completion_tokens") or usage.get(
                "output_tokens"
            )
            self.total_tokens = usage.get("total_tokens")

        for gen_list in response.generations:
            for gen in gen_list:
                info = gen.generation_info or {}
                um = info.get("usage_metadata", {})
                if um:
                    self.input_tokens = self.input_tokens or um.get(
                        "input_tokens"
                    )
                    self.output_tokens = self.output_tokens or um.get(
                        "output_tokens"
                    )
                    self.total_tokens = self.total_tokens or um.get(
                        "total_tokens"
                    )

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


llm_service = LLMService()
llm = llm_service