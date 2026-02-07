from sqlalchemy import String, Text, DateTime, func, Index, LargeBinary
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    pass


class BellyEmbedding(Base):
    __tablename__ = "belly_embedding"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_name: Mapped[str] = mapped_column(String, nullable=False)
    newt_class_name: Mapped[str] = mapped_column(String, nullable=False)

    image_sha256: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    image_bytes: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    embedding: Mapped[list[float]] = mapped_column(Vector(256), nullable=False)

    created_at: Mapped[str] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index(
            "belly_embedding_hnsw_cosine_idx",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("belly_embedding_newt_idx", "newt_class_name"),
    )
