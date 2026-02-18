import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from bot.database.utils import sha256_hex, to_vector_literal


async def insert_new_belly(
    session: AsyncSession,
    image_bytes: bytes,
    embedding: np.ndarray,
    newt_class_name: str,
    run_name: str,
) -> bool:

    sha = sha256_hex(image_bytes)
    vector_literal = to_vector_literal(embedding)


    sql = text("""
        INSERT INTO belly_embedding (
            run_name,
            newt_class_name,
            image_sha256,
            image_bytes,
            embedding
        )
        VALUES (
            :run_name,
            :newt_class_name,
            :sha,
            :image_bytes,
            CAST(:embedding AS vector)
        )
        ON CONFLICT (image_sha256) DO NOTHING
        RETURNING id;
    """)

    result = await session.execute(
        sql,
        {
            "run_name": run_name,
            "newt_class_name": newt_class_name,
            "sha": sha,
            "image_bytes": image_bytes,
            "embedding": vector_literal,
        },
    )

    await session.commit()

    return result.scalar() is not None
