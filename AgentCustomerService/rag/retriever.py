"""
RAG 检索核心

流程：
  data/faq.md → TextLoader → RecursiveCharacterTextSplitter
  → HuggingFaceEmbeddings (all-MiniLM-L6-v2) → FAISS
  → search_faq(query) → 返回最相关的文本段落

首次运行时自动下载 Embedding 模型并构建向量库（约 80MB），
后续启动直接加载本地 FAISS 持久化数据。
"""
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.core.logger import logger

# ── 路径常量 ─────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent.parent
FAQ_PATH = ROOT_DIR / "data" / "faq.md"
FAISS_INDEX_DIR = ROOT_DIR / "faiss_index"

# Embedding 模型（轻量、多语言友好，无需 GPU 也能流畅运行）
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# 切块参数
CHUNK_SIZE = 500       # 每块最多 500 字符，适配 FAQ 段落粒度
CHUNK_OVERLAP = 100    # 块间重叠 100 字符，避免关键信息被截断

# ── 单例 ─────────────────────────────────────────────────

_embedding_model: Optional[HuggingFaceEmbeddings] = None
_vectorstore: Optional[FAISS] = None


def _get_embedding_model() -> HuggingFaceEmbeddings:
    """获取 Embedding 模型单例（首次加载会下载模型）"""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"[RAG] 加载 Embedding 模型: {EMBEDDING_MODEL}")
        _embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("[RAG] Embedding 模型加载完成")
    return _embedding_model


def _load_and_split_documents() -> List[Document]:
    """加载 FAQ 文档并切块"""
    logger.info(f"[RAG] 加载文档: {FAQ_PATH}")
    loader = TextLoader(str(FAQ_PATH), encoding="utf-8")
    docs = loader.load()
    logger.info(f"[RAG] 文档加载完成 | 原始长度={len(docs[0].page_content)} 字符")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", ".", "；", ";", " "],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"[RAG] 文档切块完成 | chunks={len(chunks)}")
    return chunks


def _build_or_load_vectorstore() -> FAISS:
    """
    加载已有向量库，不存在则新建。

    FAISS 索引持久化到 FAISS_INDEX_DIR，
    后续重启直接加载，无需重新 embedding。
    """
    embeddings = _get_embedding_model()

    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # 检查是否已有持久化数据（FAISS 保存为 index.faiss + index.pkl）
    index_file = FAISS_INDEX_DIR / "index.faiss"
    if index_file.exists():
        logger.info(f"[RAG] 从本地加载 FAISS 向量库: {FAISS_INDEX_DIR}")
        vectorstore = FAISS.load_local(
            str(FAISS_INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info(f"[RAG] FAISS 向量库加载完成 | 文档数={vectorstore.index.ntotal}")
        return vectorstore

    # 构建新向量库
    logger.info("[RAG] 本地向量库不存在，开始构建…")
    chunks = _load_and_split_documents()
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )
    # 持久化到磁盘
    vectorstore.save_local(str(FAISS_INDEX_DIR))
    logger.info(f"[RAG] FAISS 向量库构建完成并持久化 | 文档数={vectorstore.index.ntotal}")
    return vectorstore


def get_vectorstore() -> FAISS:
    """获取向量库单例"""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = _build_or_load_vectorstore()
    return _vectorstore


# ── 对外接口 ──────────────────────────────────────────────

def search_faq(query: str, k: int = 3) -> str:
    """
    在 FAQ 知识库中检索与查询最相关的文本段落。

    Args:
        query:  用户的自然语言问题，如「退换货邮费谁出」「发票怎么开」
        k:      返回的最相关段落数量，默认 3

    Returns:
        格式化的检索结果字符串，包含段落内容和相关度评分。
        若无相关结果，返回提示信息。
    """
    if not query or not query.strip():
        return "（检索查询为空）"

    logger.info(f"[RAG] 知识库检索 | query={query[:100]} | k={k}")

    try:
        vectorstore = get_vectorstore()
        docs_with_scores = vectorstore.similarity_search_with_score(
            query.strip(), k=k
        )

        if not docs_with_scores:
            logger.info("[RAG] 未检索到匹配的 FAQ 条目")
            return (
                "未在知识库中检索到与您的问题直接匹配的条款。"
                "建议您联系人工客服获取更详细的解答（客服热线 400-888-0000）。"
            )

        # 格式化结果
        lines = [f"从知识库中检索到 {len(docs_with_scores)} 条相关规则：\n"]
        for i, (doc, score) in enumerate(docs_with_scores, 1):
            relevance = _score_to_label(score)
            lines.append(f"【参考条目 {i}】（相关度: {relevance}）")
            lines.append(doc.page_content.strip())
            lines.append("")

        result = "\n".join(lines)
        logger.info(
            f"[RAG] 检索完成 | result_chunks={len(docs_with_scores)} | "
            f"top_score={docs_with_scores[0][1]:.4f}"
        )
        return result

    except Exception as exc:
        logger.error(f"[RAG] 检索异常 | error={exc}")
        return f"知识库检索暂时不可用。请稍后重试或联系人工客服。错误详情: {exc}"


def _score_to_label(score: float) -> str:
    """将 FAISS L2 距离转换为可读标签（值越低越相关）"""
    # FAISS similarity_search_with_score 返回 L2 距离
    # 归一化后范围约 0~2，0=完全相同，2=完全相反
    if score < 0.4:
        return "★★★★★"
    elif score < 0.8:
        return "★★★★☆"
    elif score < 1.2:
        return "★★★☆☆"
    else:
        return "★★☆☆☆"
