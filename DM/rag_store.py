import os
import hashlib
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

# Lazy import strategy to keep the main app runnable even if DB deps are missing
try:
    from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, create_engine, func
    from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker
    from pgvector.sqlalchemy import Vector
    SQLALCHEMY_AVAILABLE = True
except Exception as import_error:  # pragma: no cover
    SQLALCHEMY_AVAILABLE = False
    _IMPORT_ERROR = import_error


Base = declarative_base() if SQLALCHEMY_AVAILABLE else None


if SQLALCHEMY_AVAILABLE:
    class Document(Base):  # type: ignore[misc]
        __tablename__ = "documents"
        id = Column(Integer, primary_key=True)
        title = Column(String(512), nullable=False)
        file_path = Column(String(2048), unique=True, nullable=False)
        content_hash = Column(String(64), nullable=False)
        created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
        chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")


    class Chunk(Base):  # type: ignore[misc]
        __tablename__ = "chunks"
        id = Column(Integer, primary_key=True)
        document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
        chunk_index = Column(Integer, nullable=False)
        text = Column(Text, nullable=False)
        token_count = Column(Integer, nullable=False, default=0)
        embedding = Column(Vector(), nullable=False)  # dim inferred from inserted vectors
        document = relationship("Document", back_populates="chunks")


@dataclass
class SearchResult:
    text: str
    document_title: str
    file_path: str
    distance: float


def _require_sqlalchemy() -> None:
    if not SQLALCHEMY_AVAILABLE:
        raise RuntimeError(f"SQLAlchemy/pgvector not available: {_IMPORT_ERROR}")


def compute_file_hash(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def simple_overlap_chunk(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    if not text:
        return []
    # Clean text: remove NUL bytes and other problematic characters
    text = text.replace('\x00', '').replace('\x01', '').replace('\x02', '').replace('\x03', '').replace('\x04', '').replace('\x05', '').replace('\x06', '').replace('\x07', '').replace('\x08', '').replace('\x0b', '').replace('\x0c', '').replace('\x0e', '').replace('\x0f', '').replace('\x10', '').replace('\x11', '').replace('\x12', '').replace('\x13', '').replace('\x14', '').replace('\x15', '').replace('\x16', '').replace('\x17', '').replace('\x18', '').replace('\x19', '').replace('\x1a', '').replace('\x1b', '').replace('\x1c', '').replace('\x1d', '').replace('\x1e', '').replace('\x1f', '')
    # Also remove other common problematic characters
    text = text.replace('\r', '\n').replace('\t', ' ')
    # Normalize whitespace
    text = ' '.join(text.split())
    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == length:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


class RagStore:
    def __init__(self, database_url: Optional[str] = None) -> None:
        _require_sqlalchemy()
        self.database_url = database_url or os.getenv("DATABASE_URL", "postgresql+psycopg://postgres@localhost:5432/dm")
        self.engine = create_engine(self.database_url, pool_pre_ping=True, future=True)
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, expire_on_commit=False, future=True)

    def ensure_schema(self) -> None:
        Base.metadata.create_all(self.engine)

    def get_document_by_path(self, session: Session, file_path: str):
        return session.query(Document).filter_by(file_path=file_path).one_or_none()

    def upsert_document(self, session: Session, title: str, file_path: str, content_hash: str):
        doc = self.get_document_by_path(session, file_path)
        if doc is None:
            doc = Document(title=title, file_path=file_path, content_hash=content_hash)
            session.add(doc)
            session.flush()
            return doc, True
        # If content changed, replace chunks
        if doc.content_hash != content_hash:
            doc.content_hash = content_hash
            # delete orphaned chunks; relationship is configured to cascade on delete-orphan
            for ch in list(doc.chunks):
                session.delete(ch)
            session.flush()
            return doc, True
        return doc, False

    def ingest_text_chunks(
        self,
        session: Session,
        document_id: int,
        chunks: Sequence[str],
        embedder: Callable[[Sequence[str]], List[List[float]]],
    ) -> int:
        embeddings = embedder(chunks)
        count = 0
        for idx, (text, emb) in enumerate(zip(chunks, embeddings)):
            session.add(
                Chunk(
                    document_id=document_id,
                    chunk_index=idx,
                    text=text,
                    token_count=len(text),
                    embedding=emb,
                )
            )
            count += 1
        return count

    def ingest_pdf(
        self,
        pdf_path: str,
        embedder: Callable[[Sequence[str]], List[List[float]]],
        chunk_size: int = 800,
        overlap: int = 120,
    ) -> Tuple[int, bool]:
        from PyPDF2 import PdfReader  # local import to keep base app light

        title = os.path.basename(pdf_path)
        content_hash = compute_file_hash(pdf_path)

        with self.SessionLocal() as session:
            doc, changed = self.upsert_document(session, title=title, file_path=pdf_path, content_hash=content_hash)
            if not changed and doc.chunks:
                session.commit()
                return len(doc.chunks), False

            # (Re)create chunks
            reader = PdfReader(pdf_path)
            full_text = "".join((page.extract_text() or "").replace('\x00', '') for page in reader.pages)
            chunks = simple_overlap_chunk(full_text, chunk_size=chunk_size, overlap=overlap)
            added = self.ingest_text_chunks(session, document_id=doc.id, chunks=chunks, embedder=embedder)
            session.commit()
            return added, True

    def search(
        self,
        query_embedding: List[float],
        document_paths: Optional[Sequence[str]] = None,
        k: int = 6,
    ) -> List[SearchResult]:
        # cosine_distance is available as a method on Vector columns
        with self.SessionLocal() as session:
            q = session.query(Chunk, Document).join(Document, Chunk.document_id == Document.id)
            if document_paths:
                q = q.filter(Document.file_path.in_(list(document_paths)))
            distance_expr = Chunk.embedding.cosine_distance(query_embedding)
            q = q.order_by(distance_expr).limit(k)
            rows = q.all()
            results: List[SearchResult] = []
            # Build results with distance values
            for chunk, doc in rows:
                # Calculate distance directly for this chunk
                dist_query = session.query(Chunk.embedding.cosine_distance(query_embedding)).filter(Chunk.id == chunk.id)
                dist_val = dist_query.scalar()
                results.append(
                    SearchResult(
                        text=chunk.text,
                        document_title=doc.title,
                        file_path=doc.file_path,
                        distance=float(dist_val) if dist_val is not None else 0.0,
                    )
                )
            return results


