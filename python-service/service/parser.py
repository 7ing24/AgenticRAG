import os
import io
import requests
import tempfile
import logging
from typing import List, Optional
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from PIL import Image
import pytesseract
from service.text_splitter import AdaptiveChunker, create_chunker
from service.table_extractor import extract_pdf, extract_docx, extract_html
from service.cleaner import clean_documents
from core.config import config

# pdf2image 可选依赖，用于扫描件 PDF 的 OCR 兜底
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

# 配置日志
logger = config.logger

# 全局变量，标记Tesseract是否可用
tesseract_available = False

# 动态配置Tesseract路径
def setup_tesseract():
    """动态配置Tesseract路径，支持多种安装位置"""
    global tesseract_available
    # 从环境变量读取Tesseract路径
    env_tesseract_path = os.getenv("TESSERACT_PATH")
    possible_paths = []

    # 如果环境变量设置了路径，优先使用
    if env_tesseract_path:
        possible_paths.append(env_tesseract_path)

    # 添加默认路径
    possible_paths.extend([
        r'C:/Program Files/Tesseract-OCR/tesseract.exe',  # 默认安装路径
        r'C:/Program Files (x86)/Tesseract-OCR/tesseract.exe',  # 32位安装路径
        r'E:/Tesseract-OCR/tesseract.exe',  # E盘安装路径
        '/usr/bin/tesseract',  # Linux/Mac
        '/usr/local/bin/tesseract',  # Linux/Mac alternative
    ])

    for path in possible_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            logger.info(f"Tesseract found at: {path}")

            # 检查语言包
            tessdata_dir = os.path.join(os.path.dirname(path), 'tessdata')
            if os.path.exists(tessdata_dir):
                logger.info(f"Tessdata directory: {tessdata_dir}")
                # 检查中文语言包
                chi_sim_path = os.path.join(tessdata_dir, 'chi_sim.traineddata')
                eng_path = os.path.join(tessdata_dir, 'eng.traineddata')

                if os.path.exists(chi_sim_path):
                    logger.info("Chinese language pack found: chi_sim.traineddata")
                else:
                    logger.warning("Chinese language pack (chi_sim.traineddata) not found!")

                if os.path.exists(eng_path):
                    logger.info("English language pack found: eng.traineddata")
                else:
                    logger.warning("English language pack (eng.traineddata) not found!")
            tesseract_available = True
            return

    logger.error("Tesseract not found in any known location!")
    # 如果找不到，尝试使用系统PATH
    try:
        pytesseract.get_tesseract_version()
        logger.info("Tesseract found in system PATH")
        tesseract_available = True
    except Exception as e:
        logger.error(f"Tesseract not found: {e}")
        logger.warning("Tesseract OCR not installed. Image OCR functionality will be disabled.")
        tesseract_available = False

# 初始化Tesseract
setup_tesseract()

class DocumentParser:
    def __init__(self):
        # 从配置管理模块读取切分配置
        chunk_size = config.CHUNK_SIZE
        chunk_overlap = config.CHUNK_OVERLAP
        min_chunk_size = config.MIN_CHUNK_SIZE
        chunk_strategy = config.CHUNK_STRATEGY

        # 获取 embeddings 实例（语义切分需要）
        embeddings = None
        if chunk_strategy == "semantic":
            try:
                from core.vector_store import vector_store_manager
                embeddings = vector_store_manager.embeddings
            except Exception:
                logger.warning(
                    "Cannot access shared embeddings, "
                    "will fall back to structural chunking"
                )

        # 使用自适应切分器
        self.chunker = create_chunker({
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "min_chunk_size": min_chunk_size,
            "strategy": chunk_strategy,
            "semantic_breakpoint_type": config.SEMANTIC_BREAKPOINT_TYPE,
            "semantic_breakpoint_amount": config.SEMANTIC_BREAKPOINT_AMOUNT,
            "semantic_buffer_size": config.SEMANTIC_BUFFER_SIZE,
        }, embeddings=embeddings)

        # 父子块切分器（父块用语义切分，需要 embeddings）
        from service.parent_child_chunker import ParentChildChunker
        self.parent_child_chunker = ParentChildChunker(
            embeddings=embeddings,
            child_chunk_size=config.CHILD_CHUNK_SIZE,
            child_chunk_overlap=config.CHILD_CHUNK_OVERLAP,
        )

        logger.info(f"DocumentParser initialized with strategy: {chunk_strategy}, chunk_size: {chunk_size}")

    def parse_parent_child(self, file_path: str):
        """父子块模式：返回 (parent_docs, child_docs)"""
        raw_docs = self.parse(file_path, skip_chunk=True)
        return self.parent_child_chunker.split_documents(raw_docs)

    # =========================================================================
    # OCR 相关方法
    # =========================================================================

    LANG_CONFIGS = ['chi_sim+eng', 'chi_sim', 'eng', 'chi_tra+eng']
    OCR_MIN_TEXT_LENGTH = 50  # 单页少于这些字符视为空白页

    def _ocr_image(self, image_path: str) -> List[Document]:
        """对单张图片执行 OCR，返回文档列表"""
        try:
            logger.info(f"Processing image file: {image_path}")

            if not tesseract_available:
                logger.warning("Tesseract OCR is not available")
                return [Document(
                    page_content="图片OCR处理失败: Tesseract OCR未安装",
                    metadata={"source": image_path, "page": 1,
                              "file_type": "image", "error": "Tesseract not available"}
                )]

            image = Image.open(image_path)
            ocr_text, ocr_errors = self._ocr_from_image(image)

            if not ocr_text or not ocr_text.strip():
                ocr_text = "图片中未识别到文字"
                logger.warning("No text detected in image")

            return [Document(
                page_content=ocr_text,
                metadata={
                    "source": image_path, "page": 1, "file_type": "image",
                    "ocr_errors": ocr_errors if ocr_errors else None,
                }
            )]
        except Exception as e:
            logger.error(f"Failed to OCR image {image_path}: {e}")
            return [Document(
                page_content=f"图片OCR处理失败: {str(e)}",
                metadata={"source": image_path, "page": 1,
                          "file_type": "image", "error": str(e)}
            )]

    def _ocr_from_image(self, image: Image.Image) -> tuple:
        """对 PIL Image 执行 OCR，返回 (text, errors)"""
        ocr_text = ""
        ocr_errors = []

        # 预处理：转灰度
        if image.mode != 'L':
            image = image.convert('L')

        # 预处理：缩放到合理尺寸
        max_size = 2000
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            logger.info(f"Resized image to {new_size}")

        for lang in self.LANG_CONFIGS:
            try:
                logger.info(f"Trying OCR with language: {lang}")
                text = pytesseract.image_to_string(
                    image, lang=lang, config='--psm 3 --oem 3'
                )
                if text and text.strip():
                    ocr_text = text.strip()
                    logger.info(
                        f"OCR successful with {lang}, "
                        f"text length: {len(ocr_text)}"
                    )
                    break
                else:
                    logger.warning(f"No text detected with: {lang}")
            except Exception as e:
                ocr_errors.append(f"{lang}: {e}")
                logger.warning(f"OCR {lang} failed: {e}")

        if not ocr_text:
            try:
                logger.info("Trying OCR with default settings")
                ocr_text = pytesseract.image_to_string(image).strip()
            except Exception as e:
                logger.error(f"Default OCR failed: {e}")

        return ocr_text, ocr_errors

    def _is_scanned_pdf(self, documents: List[Document]) -> bool:
        """判断 PDF 是否为扫描件（图片型 PDF，PyPDFLoader 拿不到文本）

        规则：超过半数页面文本量 < OCR_MIN_TEXT_LENGTH 且总文本 < 200 字符
        """
        if not documents:
            return True

        total_text = "".join(d.page_content for d in documents)
        blank_pages = sum(
            1 for d in documents if len(d.page_content.strip()) < self.OCR_MIN_TEXT_LENGTH
        )
        blank_ratio = blank_pages / len(documents)

        return blank_ratio > 0.5 and len(total_text) < 200

    def _ocr_pdf(self, pdf_path: str) -> Optional[List[Document]]:
        """将 PDF 逐页转为图片后 OCR，作为扫描件兜底"""
        if not tesseract_available:
            logger.warning("Tesseract not available, cannot OCR PDF")
            return None

        if not PDF2IMAGE_AVAILABLE:
            logger.warning(
                "pdf2image not installed, cannot OCR scanned PDF. "
                "Install: pip install pdf2image && apt install poppler-utils"
            )
            return None

        try:
            logger.info(f"Converting PDF pages to images for OCR: {pdf_path}")
            images = convert_from_path(pdf_path, dpi=200)
            logger.info(f"PDF has {len(images)} pages")
        except Exception as e:
            logger.error(f"Failed to convert PDF to images: {e}")
            return None

        documents = []
        for i, image in enumerate(images):
            logger.info(f"OCR page {i+1}/{len(images)}")
            ocr_text, ocr_errors = self._ocr_from_image(image)
            if not ocr_text or not ocr_text.strip():
                ocr_text = f"第{i+1}页未识别到文字"
            documents.append(Document(
                page_content=ocr_text,
                metadata={
                    "source": pdf_path, "page": i + 1,
                    "file_type": "scanned_pdf",
                    "ocr_errors": ocr_errors if ocr_errors else None,
                }
            ))

        logger.info(f"Scanned PDF OCR complete: {len(documents)} pages")
        return documents

    def parse(self, file_path: str, skip_chunk: bool = False) -> List[Document]:
        """
        根据文件扩展名选择合适的加载器解析文档，并切分文本。
        支持本地路径和 HTTP/HTTPS URL。
        skip_chunk=True 时只加载不切分，供父子块模式复用。
        """
        # 检测是否为 URL(支持带或不带协议头)
        is_url = file_path.startswith(('http://', 'https://')) or \
                 (not os.path.exists(file_path) and
                  ('clouddn.com' in file_path or 'aliyuncs.com' in file_path or '/' in file_path))
        temp_file = None

        try:
            target_path = file_path

            # 如果是 URL，先下载到临时文件
            if is_url:
                try:
                    # 如果没有协议头，添加 https://
                    download_url = file_path
                    if not download_url.startswith(('http://', 'https://')):
                        download_url = 'https://' + download_url

                    response = requests.get(download_url, stream=True, timeout=30)
                    response.raise_for_status()

                    # 推断扩展名，优先从 URL 获取，如果没有则尝试从 Content-Type 或 Content-Disposition 获取
                    # 简单起见，这里假设 URL 包含扩展名
                    ext = os.path.splitext(file_path)[1].lower()
                    if not ext:
                        # 尝试从 Content-Type 推断
                        content_type = response.headers.get('Content-Type', '').lower()
                        if 'pdf' in content_type:
                            ext = '.pdf'
                        elif 'word' in content_type:
                            ext = '.docx'
                        elif 'markdown' in content_type:
                            ext = '.md'
                        elif 'image' in content_type:
                            # 图片类型
                            if 'png' in content_type:
                                ext = '.png'
                            elif 'jpeg' in content_type or 'jpg' in content_type:
                                ext = '.jpg'
                            elif 'gif' in content_type:
                                ext = '.gif'
                            elif 'bmp' in content_type:
                                ext = '.bmp'
                            else:
                                ext = '.png'  # 默认使用png
                        else:
                            ext = '.txt'

                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
                    for chunk in response.iter_content(chunk_size=8192):
                        temp_file.write(chunk)
                    temp_file.close()
                    target_path = temp_file.name
                except Exception as e:
                    raise Exception(f"Failed to download file from URL: {e}")
            else:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File not found: {file_path}")

            ext = os.path.splitext(target_path)[1].lower()

            if ext == '.pdf':
                if config.ENABLE_TABLE_EXTRACTION:
                    documents = extract_pdf(target_path)
                else:
                    from langchain_community.document_loaders import PyPDFLoader
                    documents = PyPDFLoader(target_path).load()
                # 扫描件 PDF 兜底：对图片型 PDF（无文本层）走 OCR
                if self._is_scanned_pdf(documents):
                    logger.info(f"Detected scanned PDF, falling back to OCR: {target_path}")
                    ocr_docs = self._ocr_pdf(target_path)
                    if ocr_docs:
                        documents = ocr_docs
            elif ext == '.docx':
                if config.ENABLE_TABLE_EXTRACTION:
                    documents = extract_docx(target_path)
                else:
                    from langchain_community.document_loaders import Docx2txtLoader
                    documents = Docx2txtLoader(target_path).load()
            elif ext == '.txt':
                loader = TextLoader(target_path, encoding='utf-8')
                documents = loader.load()
            elif ext == '.md':
                loader = TextLoader(target_path, encoding='utf-8')
                documents = loader.load()
            elif ext in ['.html', '.htm']:
                documents = extract_html(target_path)
            elif ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif']:
                documents = self._ocr_image(target_path)
            else:
                raise ValueError(f"Unsupported file type: {ext}")

            # 清洗（加载后、切分前统一执行）
            if config.ENABLE_CLEANING:
                documents = clean_documents(documents)

            # 使用自适应切分器进行文档切分
            if skip_chunk:
                return documents
            chunks = self.chunker.split_documents(documents)
            return chunks

        finally:
            # 清理临时文件
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except Exception:
                    pass

