from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from bot.database.utils import to_vector_literal
from bot.domain.newt_match import NewtMatch


async def find_best_match(
    session: AsyncSession,
    query_emb: np.ndarray,
    top_k: int = 5,
) -> List[NewtMatch]:

    vector_literal = to_vector_literal(query_emb)

    sql = text("""
        WITH best_per_class AS (
            SELECT DISTINCT ON (newt_class_name)
                newt_class_name,
                (embedding <=> CAST(:embedding AS vector)) AS distance
            FROM belly_embedding
            ORDER BY newt_class_name, distance
        )
        SELECT
            newt_class_name,
            1.0 - distance AS similarity
        FROM best_per_class
        ORDER BY distance
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
        NewtMatch(
            class_name=row["newt_class_name"],
            similarity=float(row["similarity"]),
        )
        for row in rows
    ]
