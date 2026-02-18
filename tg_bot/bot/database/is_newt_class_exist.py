from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def is_newt_class_exist(
    session: AsyncSession,
    newt_class: str,
) -> bool:
    sql = """
    SELECT 1
    FROM belly_embedding
    WHERE newt_class_name = :newt_class
    LIMIT 1;
    """

    result = await session.execute(
        text(sql),
        {"newt_class": newt_class},
    )

    return result.scalar() is not None
