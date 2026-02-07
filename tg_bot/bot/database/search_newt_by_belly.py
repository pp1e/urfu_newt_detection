from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import psycopg
from psycopg.rows import dict_row


@dataclass(frozen=True)
class Match:
    newt_class_name: str
    similarity: float


def _to_vector_literal(vec: np.ndarray) -> str:
    return "[" + ",".join(f"{float(x):.8f}" for x in vec) + "]"


async def find_best_match(
    database_url: str,
    query_emb: np.ndarray,
    *,
    top_k: int = 5,
) -> List[Match]:
    vec_literal = _to_vector_literal(query_emb)

    sql = """
    SELECT
        newt_class_name,
        1.0 - (embedding <=> %s::vector) AS similarity
    FROM belly_embedding
    ORDER BY embedding <=> %s::vector
    LIMIT %s;
    """

    async with await psycopg.AsyncConnection.connect(database_url, row_factory=dict_row) as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql, (vec_literal, vec_literal, top_k))
            rows = await cur.fetchall()

    return [Match(newt_class_name=r["newt_class_name"], similarity=float(r["similarity"])) for r in rows]
