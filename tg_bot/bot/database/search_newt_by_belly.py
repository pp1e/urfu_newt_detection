from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from bot.database.utils import to_vector_literal


@dataclass(frozen=True)
class Match:
    newt_class_name: str
    similarity: float


async def find_best_match(
    session: AsyncSession,
    query_emb: np.ndarray,
    top_k: int = 5,
) -> List[Match]:

    vector_literal = to_vector_literal(query_emb)

    sql = text("""
        SELECT
            newt_class_name,
            1.0 - (embedding <=> CAST(:embedding AS vector)) AS similarity
        FROM belly_embedding
        ORDER BY embedding <=> CAST(:embedding AS vector)
        LIMIT :top_k;
    """)

    result = await session.execute(
        sql,
        {
            "embedding": vector_literal,
            "top_k": top_k,
        },
    )

    rows = result.mappings().all()

    return [
        Match(
            newt_class_name=row["newt_class_name"],
            similarity=float(row["similarity"]),
        )
        for row in rows
    ]
