"""
RAG 检索核心（ChromaDB 版本）

架构演进：FAISS + all-MiniLM-L6-v2 → ChromaDB + BGE-small-zh-v1.5

流程：
  data/faq.md → TextLoader → RecursiveCharacterTextSplitter
  → HuggingFaceEmbeddings (bge-small-zh-v1.5) → ChromaDB (Docker Server)
  → search_faq(query, user_profile=None) → 返回最相关的文本段落

核心原则：
  1. Client/Server 架构：ChromaDB 通过 Docker 独立部署，
     Python 后端仅通过 chromadb.HttpClient 连接，绝不本地实例化。
  2. 所有阻塞操作均通过 asyncio.to_thread() 包装，
     杜绝阻塞 FastAPI 事件循环。
  3. Metadata Filtering：文档带业务标签入库，检索时支持
     先标签过滤 + 后语义检索的混合策略。
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── HuggingFace 网络优化：优先使用本地缓存 ──
# 必须在加载 HuggingFaceEmbeddings 之前设置，否则会因 HF 不可达而超时
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from app.core.logger import logger
from services.retry import (
    chromadb_breaker,
    retry_on_network_error,
)

# ═══════════════════════════════════════════════════════════════
# 路径常量
# ═══════════════════════════════════════════════════════════════

ROOT_DIR = Path(__file__).resolve().parent.parent
FAQ_PATH = ROOT_DIR / "data" / "faq.md"

# ═══════════════════════════════════════════════════════════════
# 配置（环境变量优先，代码常量兜底）
# ═══════════════════════════════════════════════════════════════

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8001"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "cs_faq")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# ═══════════════════════════════════════════════════════════════
# FAQ 文档元数据标签规则
# ═══════════════════════════════════════════════════════════════

# 章节标题 → policy_type 映射（用于自动打标）
SECTION_POLICY_MAP: Dict[str, str] = {
    "退换货": "return_exchange",
    "发票": "invoice",
    "物流": "logistics",
    "会员": "membership",
    "售后": "after_sales",
}

# 会员等级关键词 → 适用等级列表
LEVEL_KEYWORDS: Dict[str, List[str]] = {
    "普通会员": ["basic"],
    "银卡": ["silver"],
    "金卡": ["gold"],
    "钻石": ["diamond"],
    "全场景包邮": ["gold", "diamond"],
    "优先发货": ["diamond"],
    "专属客服": ["gold", "diamond"],
    "30 天无理由": ["diamond"],
    "包邮门槛 99": ["basic"],
    "包邮门槛降至 69": ["silver"],
}

# ═══════════════════════════════════════════════════════════════
# 当前请求级用户画像（线程安全上下文）
# ═══════════════════════════════════════════════════════════════

_current_request_profile: Optional[Dict[str, Any]] = None


def set_current_profile(profile: Optional[Dict[str, Any]]) -> None:
    """
    设置当前请求的用户画像上下文。

    由 agent/core.py 在每次 Agent.run() 前调用，
    供 search_faq() 自动读取以实现个性化的业务标签过滤。
    """
    global _current_request_profile
    _current_request_profile = profile


def get_current_profile() -> Optional[Dict[str, Any]]:
    """获取当前请求的用户画像上下文。"""
    return _current_request_profile


# ═══════════════════════════════════════════════════════════════
# 元数据标签推断
# ═══════════════════════════════════════════════════════════════

def _infer_policy_type(text: str) -> Optional[str]:
    """根据文档内容推断业务政策类型。"""
    for keyword, ptype in SECTION_POLICY_MAP.items():
        if keyword in text:
            return ptype
    return None


def _infer_member_levels(text: str) -> Dict[str, bool]:
    """
    根据文档内容推断适用的会员等级。

    返回形如 {"level_gold": True, "level_diamond": True} 的 dict，
    每个等级独立成字段——兼容 ChromaDB 0.5.x 的元数据类型限制
    （0.5.x 仅支持 str / int / float / bool，不支持 list）。
    """
    levels: Dict[str, bool] = {}
    for keyword, level_list in LEVEL_KEYWORDS.items():
        if keyword in text:
            for lv in level_list:
                levels[f"level_{lv}"] = True
    return levels


def _build_metadata_for_chunk(text: str, section: str = "") -> Dict[str, Any]:
    """为一个文档块构建元数据字典。"""
    # 默认 policy_type，确保 metadata 非空（ChromaDB 要求至少一个字段）
    policy_type = _infer_policy_type(text) or _infer_policy_type(section) or "general"

    meta: Dict[str, Any] = {"policy_type": policy_type}

    # 将等级信息展开为独立布尔字段（level_basic / level_silver / level_gold / level_diamond）
    level_flags = _infer_member_levels(text)
    meta.update(level_flags)

    return meta


# ═══════════════════════════════════════════════════════════════
# 用户画像 → ChromaDB where 过滤器
# ═══════════════════════════════════════════════════════════════

def build_where_filter(user_profile: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    将用户画像 dict 转换为 ChromaDB where 过滤条件。

    ChromaDB where 语法：
      - 等值:  {"policy_type": "membership"}
      - $and:  {"$and": [{"policy_type": "membership"}, {"applicable_levels": {"$in": ["gold"]}}]}
      - $or:   {"$or": [...]}
      - 空值: 返回 None 表示不过滤

    Args:
        user_profile: 用户画像 JSON，如 {"card_type": "times_card", "level": "gold"}

    Returns:
        ChromaDB Where 字典，None 表示不过滤
    """
    if not user_profile:
        return None

    conditions: List[Dict[str, Any]] = []

    # 会员等级 → level_* 布尔字段过滤
    level = user_profile.get("level") or user_profile.get("会员等级")
    if level:
        level_map = {
            "basic": "level_basic",
            "silver": "level_silver",
            "gold": "level_gold",
            "diamond": "level_diamond",
            "普通会员": "level_basic",
            "普通": "level_basic",
            "银卡": "level_silver",
            "金卡": "level_gold",
            "钻石": "level_diamond",
        }
        normalized = level_map.get(str(level).strip(), None)
        if normalized:
            # ChromaDB 0.5.x 仅支持 str / int / float / bool 元数据值
            # level_* 字段为布尔值，直接等值匹配
            conditions.append({normalized: True})

    # card_type 过滤（如果文档标注了 card_type）
    card_type = user_profile.get("card_type")
    if card_type:
        conditions.append({"card_type": str(card_type)})

    # policy_type 可以根据用户偏好过滤（如用户偏好物流相关内容）
    preferred_policy = user_profile.get("policy_preference") or user_profile.get("关注领域")
    if preferred_policy:
        conditions.append({"policy_type": str(preferred_policy)})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# ═══════════════════════════════════════════════════════════════
# VectorStoreService — 异步 ChromaDB 客户端
# ═══════════════════════════════════════════════════════════════

class VectorStoreService:
    """
    向量存储服务 —— 封装 ChromaDB + Embedding 模型。

    设计要点：
      - 通过 chromadb.HttpClient 连接 Docker 容器（绝不本地实例化）
      - 所有阻塞 I/O 均通过 asyncio.to_thread() 包装
      - 内置 tenacity 重试机制，应对网络抖动
      - 支持 metadata 过滤（Chroma where 子句）
      - 单例模式
    """

    _instance: Optional["VectorStoreService"] = None

    def __init__(self) -> None:
        self._embedding_model: Optional[HuggingFaceEmbeddings] = None
        self._chroma_client: Optional[chromadb.HttpClient] = None
        self._collection: Optional[chromadb.Collection] = None
        self._initialized: bool = False

    # ── 单例 ─────────────────────────────────────────────────

    @classmethod
    def get_instance(cls) -> "VectorStoreService":
        """获取单例实例。"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── 初始化 ───────────────────────────────────────────────

    async def initialize(
        self,
        host: str = CHROMA_HOST,
        port: int = CHROMA_PORT,
        collection_name: str = CHROMA_COLLECTION,
    ) -> None:
        """
        异步初始化服务（应在 FastAPI lifespan 启动阶段调用）。

        1. 加载本地 BGE Embedding 模型（CPU 推理）
        2. 连接 Docker ChromaDB 服务
        3. 获取或创建 Collection；若为空则自动从 FAQ 构建
        """
        if self._initialized:
            return

        import asyncio

        logger.info(
            f"[VectorStore] ═══ 初始化向量存储服务 ═══\n"
            f"  ChromaDB:  {host}:{port}\n"
            f"  Embedding: {EMBEDDING_MODEL} ({EMBEDDING_DEVICE})\n"
            f"  Collection: {collection_name}"
        )

        # 1. 加载 Embedding 模型（在线程池中执行，避免阻塞）
        self._embedding_model = await asyncio.to_thread(self._load_embedding_model)

        # 2. 连接 ChromaDB 服务
        self._chroma_client = await asyncio.to_thread(
            self._connect_chromadb, host, port
        )

        # 3. 获取或创建 Collection
        self._collection = await asyncio.to_thread(
            self._get_or_create_collection, collection_name
        )

        # 4. 若 Collection 为空，从 FAQ 文档构建索引
        count = await self.count()
        if count == 0:
            logger.info("[VectorStore] Collection 为空，开始从 FAQ 文档构建索引…")
            chunks = _load_and_split_documents()
            await self.add_documents(chunks)
            logger.info(f"[VectorStore] FAQ 索引构建完成 | documents={len(chunks)}")
        else:
            logger.info(f"[VectorStore] Collection 已存在 | documents={count}")

        self._initialized = True
        logger.info("[VectorStore] ═══ 向量存储服务就绪 ═══")

    def _load_embedding_model(self) -> HuggingFaceEmbeddings:
        """加载 Embedding 模型（CPU 推理，仅使用本地缓存）。"""
        logger.info(f"[VectorStore] 加载 Embedding 模型: {EMBEDDING_MODEL}")
        model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("[VectorStore] Embedding 模型加载完成")
        return model

    def _connect_chromadb(self, host: str, port: int) -> chromadb.HttpClient:
        """
        连接 Docker ChromaDB 服务。

        ⚠️ 仅通过 HTTP Client 连接远程服务，绝不创建本地 embedded Client。
        """
        logger.info(f"[VectorStore] 连接 ChromaDB 服务 | host={host}:{port}")
        client = chromadb.HttpClient(
            host=host,
            port=port,
            settings=ChromaSettings(
                anonymized_telemetry=False,
            ),
        )
        # 心跳检查
        heartbeat = client.heartbeat()
        logger.info(f"[VectorStore] ChromaDB 连接成功 | heartbeat={heartbeat}")
        return client

    def _get_or_create_collection(self, collection_name: str) -> chromadb.Collection:
        """
        获取或创建 Collection。

        使用 L2 距离度量（与 FAISS IndexFlatL2 行为一致），
        配合 normalize_embeddings=True 时等价于余弦相似度。
        """
        logger.info(f"[VectorStore] 获取/创建 Collection | name={collection_name}")
        collection = self._chroma_client.get_or_create_collection(  # type: ignore[union-attr]
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # 余弦距离（embedding 已归一化时等价 L2）
        )
        logger.info(f"[VectorStore] Collection 就绪 | name={collection_name}")
        return collection

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    # ── 文档管理 ─────────────────────────────────────────────

    async def add_documents(
        self,
        documents: List[Document],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        异步批量添加文档（含自动 metadata 打标）。

        Args:
            documents: LangChain Document 列表
            metadatas: 显式指定的元数据列表（可选，None 则自动推断）
            ids: 文档 ID 列表（可选，None 则自动生成）

        Returns:
            文档 ID 列表
        """
        import asyncio
        import uuid

        if not documents:
            return []

        # 提取文本和元数据
        texts = [doc.page_content for doc in documents]

        # 元数据：显式指定 > 自动推断
        if metadatas is None:
            metadatas = []
            for doc in documents:
                section = doc.metadata.get("source", "")
                meta = _build_metadata_for_chunk(doc.page_content, section)
                metadatas.append(meta)

        if ids is None:
            ids = [f"faq_{uuid.uuid4().hex[:12]}" for _ in documents]

        # ── 断路器检查 ──
        if chromadb_breaker.is_open:
            raise ConnectionError(
                f"ChromaDB 断路器开路，拒绝写入请求（连续失败 {chromadb_breaker._failure_count} 次）"
            )

        logger.info(f"[VectorStore] 添加文档 | count={len(texts)}")

        # 在线程池中执行阻塞操作（含自动重试）
        try:
            result = await asyncio.to_thread(
                self._add_to_collection,
                texts=texts,
                metadatas=metadatas,
                ids=ids,
            )
            chromadb_breaker.record_success()
            return result
        except Exception:
            chromadb_breaker.record_failure()
            raise

    @retry_on_network_error
    def _add_to_collection(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
    ) -> List[str]:
        """同步执行：embed + 写入 ChromaDB（含自动重试）。"""
        # 必须确保 _embedding_model 和 _collection 已初始化
        if self._embedding_model is None or self._collection is None:
            raise RuntimeError("VectorStoreService 未初始化，请先调用 initialize()")

        # 批量生成 embedding
        embeddings = self._embedding_model.embed_documents(texts)

        # 写入 ChromaDB
        self._collection.add(
            ids=ids,
            embeddings=embeddings,  # type: ignore[arg-type]
            documents=texts,
            metadatas=metadatas,  # type: ignore[arg-type]
        )
        logger.info(f"[VectorStore] 文档写入完成 | count={len(ids)}")
        return ids

    async def count(self) -> int:
        """异步获取文档总数。"""
        import asyncio

        if self._collection is None:
            return 0
        return await asyncio.to_thread(self._collection.count)

    # ── 检索 ─────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        k: int = 3,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Document], List[float]]:
        """
        异步语义检索（核心方法）。

        流程：
          1. 断路器检查（若开路则快速失败）
          2. 将 query 在线程池中 embedding
          3. 向 ChromaDB 发起 HTTP 查询（where 过滤 + 语义排序）
          4. 记录成功/失败（断路器的输入信号）

        Args:
            query:        用户的自然语言查询
            k:            返回的最相关文档数
            where_filter: ChromaDB where 过滤条件（业务标签预过滤）

        Returns:
            (documents, distances) 元组
        """
        import asyncio

        if self._embedding_model is None or self._collection is None:
            raise RuntimeError("VectorStoreService 未初始化，请先调用 initialize()")

        # ── 断路器检查 ──
        if chromadb_breaker.is_open:
            raise ConnectionError(
                f"ChromaDB 断路器开路，拒绝查询请求（请等待 {chromadb_breaker.cooldown_seconds}s 冷却）"
            )

        logger.info(
            f"[VectorStore] 检索请求 | query={query[:100]} | k={k} | "
            f"filter={json.dumps(where_filter, ensure_ascii=False) if where_filter else 'none'}"
        )

        try:
            # 1. 生成查询向量（线程池）
            query_embedding = await asyncio.to_thread(
                self._embedding_model.embed_query, query.strip()
            )

            # 2. ChromaDB 检索（线程池，含自动重试）
            results = await asyncio.to_thread(
                self._query_collection,
                query_embedding=query_embedding,
                k=k,
                where_filter=where_filter,
            )

            # 3. 构造返回结果
            docs: List[Document] = []
            distances: List[float] = []

            if results and results.get("ids") and results["ids"][0]:
                ids_list: List[str] = results["ids"][0]
                docs_list: List[str] = results.get("documents", [[]])[0] or []
                metas_list: List[dict] = results.get("metadatas", [[]])[0] or []
                dist_list: List[float] = results.get("distances", [[]])[0] or []

                for i, doc_id in enumerate(ids_list):
                    text = docs_list[i] if i < len(docs_list) else ""
                    meta = metas_list[i] if i < len(metas_list) else {}
                    dist = dist_list[i] if i < len(dist_list) else float("inf")

                    doc = Document(
                        id=doc_id,
                        page_content=text,
                        metadata=meta,
                    )
                    docs.append(doc)
                    distances.append(dist)

            # ── 记录成功 ──
            chromadb_breaker.record_success()

            logger.info(
                f"[VectorStore] 检索完成 | hits={len(docs)} | "
                f"top_dist={distances[0]:.4f}" if distances else "[VectorStore] 检索完成 | hits=0"
            )
            return docs, distances

        except Exception:
            # ── 记录失败（网络错误等可重试异常已在 _query_collection 层面重试） ──
            chromadb_breaker.record_failure()
            raise

    @retry_on_network_error
    def _query_collection(
        self,
        query_embedding: List[float],
        k: int,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """同步执行：向 ChromaDB 发起查询（含自动重试）。"""
        if self._collection is None:
            raise RuntimeError("VectorStoreService 未初始化")

        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter is not None:
            kwargs["where"] = where_filter

        return self._collection.query(**kwargs)

    # ── 删除 ─────────────────────────────────────────────────

    async def delete_by_ids(self, ids: List[str]) -> None:
        """异步删除指定文档。"""
        import asyncio

        if self._collection is None or not ids:
            return
        await asyncio.to_thread(self._collection.delete, ids=ids)
        logger.info(f"[VectorStore] 删除文档 | count={len(ids)}")

    async def delete_collection(self) -> None:
        """异步删除整个 Collection（谨慎使用）。"""
        import asyncio

        if self._chroma_client is None or self._collection is None:
            return
        name = self._collection.name
        await asyncio.to_thread(self._chroma_client.delete_collection, name)
        self._collection = None
        logger.info(f"[VectorStore] Collection 已删除 | name={name}")


# ═══════════════════════════════════════════════════════════════
# 文档加载与切块（与旧版兼容）
# ═══════════════════════════════════════════════════════════════

def _load_and_split_documents() -> List[Document]:
    """加载 FAQ 文档并切块（同步，仅在初始化时调用一次）。"""
    logger.info(f"[RAG] 加载文档: {FAQ_PATH}")
    loader = TextLoader(str(FAQ_PATH), encoding="utf-8")
    docs = loader.load()
    logger.info(f"[RAG] 文档加载完成 | 原始长度={len(docs[0].page_content)} 字符")

    # 先按章节分割，为每个大段打标，再用 splitter 切块
    raw_text = docs[0].page_content
    current_section = ""
    section_chunks: List[tuple] = []  # (text, section_name)

    for line in raw_text.split("\n"):
        # 检测章节标题（## 开头）
        for keyword, ptype in SECTION_POLICY_MAP.items():
            if f"## {keyword}" in line or f"### {keyword}" in line:
                current_section = keyword
                break
        section_chunks.append((line, current_section))

    # 重建带章节标记的文本（每行附加不可见的章节上下文）
    enriched_text = "\n".join(
        f"{line}" for line, _ in section_chunks
    )

    # 使用 splitter 切块
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", ".", "；", ";", " "],
        length_function=len,
    )
    # 重新加载为 Document 并切块
    from langchain_core.documents import Document as LCDocument
    enriched_doc = LCDocument(page_content=enriched_text)
    chunks = splitter.split_documents([enriched_doc])

    # 为每个 chunk 推断章节（基于其文本内容匹配章节关键词）
    for chunk in chunks:
        detected_section = ""
        for keyword in SECTION_POLICY_MAP:
            if keyword in chunk.page_content:
                detected_section = keyword
                break
        if detected_section:
            chunk.metadata["source"] = detected_section

    logger.info(f"[RAG] 文档切块完成 | chunks={len(chunks)}")
    return chunks


# ═══════════════════════════════════════════════════════════════
# 对外接口（向后兼容）
# ═══════════════════════════════════════════════════════════════

async def get_vector_service() -> VectorStoreService:
    """获取（并惰性初始化）向量存储服务单例。"""
    service = VectorStoreService.get_instance()
    if not service.is_initialized:
        await service.initialize()
    return service


async def search_faq(
    query: str,
    k: int = 3,
    user_profile: Optional[Dict[str, Any]] = None,
) -> str:
    """
    在 FAQ 知识库中检索与查询最相关的文本段落。

    支持基于用户画像的业务标签预过滤（ChromaDB where 子句）：
      - 若传入 user_profile，先按画像中的 card_type / level 等字段过滤，
        再在过滤结果中执行语义检索。
      - 若 user_profile 为 None 但已通过 set_current_profile() 设置上下文，
        则自动从上下文读取。
      - 若画像不存在，则执行全库语义检索（退化行为，向后兼容）。

    Args:
        query:        用户的自然语言问题，如「退换货邮费谁出」「发票怎么开」
        k:            返回的最相关段落数量，默认 3
        user_profile: 用户画像 dict，如 {"card_type": "times_card", "level": "gold"}
                      通常来自 SQLite user_memory.facts 字段

    Returns:
        格式化的检索结果字符串，包含段落内容和相关度评分。
    """
    if not query or not query.strip():
        return "（检索查询为空）"

    # 优先使用显式传入的 profile，其次从上下文读取
    profile = user_profile or get_current_profile()

    logger.info(
        f"[RAG] 知识库检索 | query={query[:100]} | k={k} | "
        f"profile={'yes' if profile else 'none'}"
    )

    try:
        service = await get_vector_service()

        # ── 构建 where 过滤器 ──
        where_filter = build_where_filter(profile)

        # ── 第一轮：带标签过滤的检索 ──
        if where_filter is not None:
            logger.info(
                f"[RAG] 先标签过滤后检索 | filter={json.dumps(where_filter, ensure_ascii=False)}"
            )
            docs, distances = await service.search(
                query=query.strip(),
                k=k,
                where_filter=where_filter,
            )

            # 若过滤后结果不足，补充无过滤的全库检索结果
            if len(docs) < k:
                logger.info(
                    f"[RAG] 过滤结果不足 ({len(docs)}/{k})，补充全库检索"
                )
                remaining = k - len(docs)
                docs_extra, dist_extra = await service.search(
                    query=query.strip(),
                    k=remaining,
                    where_filter=None,
                )
                # 去重合并
                seen_texts = {doc.page_content for doc in docs}
                for doc, dist in zip(docs_extra, dist_extra):
                    if doc.page_content not in seen_texts:
                        docs.append(doc)
                        distances.append(dist)
                        seen_texts.add(doc.page_content)
        else:
            # 无 profile → 全库语义检索
            docs, distances = await service.search(
                query=query.strip(),
                k=k,
                where_filter=None,
            )

        # ── 格式化结果 ──
        if not docs:
            logger.info("[RAG] 未检索到匹配的 FAQ 条目")
            return (
                "未在知识库中检索到与您的问题直接匹配的条款。"
                "建议您联系人工客服获取更详细的解答（客服热线 400-888-0000）。"
            )

        lines = [f"从知识库中检索到 {len(docs)} 条相关规则：\n"]
        for i, (doc, dist) in enumerate(zip(docs, distances), 1):
            relevance = _distance_to_label(dist)
            lines.append(f"【参考条目 {i}】（相关度: {relevance}）")

            # 展示元数据标签（便于调试/审计）
            if doc.metadata:
                tags = ", ".join(
                    f"{k}={v}" for k, v in doc.metadata.items()
                )
                lines.append(f"[标签: {tags}]")

            lines.append(doc.page_content.strip())
            lines.append("")

        result = "\n".join(lines)
        top_dist = distances[0] if distances else float("inf")
        logger.info(
            f"[RAG] 检索完成 | result_chunks={len(docs)} | top_dist={top_dist:.4f}"
        )
        return result

    except ConnectionError as exc:
        logger.error(f"[RAG] ChromaDB 连接失败 | error={exc}")
        return (
            "知识库服务暂时不可用（向量数据库连接异常）。"
            "请稍后重试或联系人工客服。"
        )
    except Exception as exc:
        logger.error(f"[RAG] 检索异常 | error={exc}")
        return (
            "知识库检索暂时不可用。请稍后重试或联系人工客服。"
            f"错误详情: {exc}"
        )


def _distance_to_label(distance: float) -> str:
    """
    将余弦距离（0~2）转换为可读的相关度标签。

    ChromaDB 使用 cosine 距离（值域 0~2）：
      - 0     = 完全相同（归一化后两个向量重合）
      - 1     = 正交（无相关性）
      - 2     = 完全相反
    """
    if distance < 0.2:
        return "★★★★★"
    elif distance < 0.5:
        return "★★★★☆"
    elif distance < 1.0:
        return "★★★☆☆"
    elif distance < 1.5:
        return "★★☆☆☆"
    else:
        return "★☆☆☆☆"
