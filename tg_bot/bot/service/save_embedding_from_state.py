import numpy as np
from aiogram.fsm.context import FSMContext
from sqlalchemy.ext.asyncio import AsyncSession

from bot.database.insert_new_belly import insert_new_belly


async def save_embedding_from_state(
    session: AsyncSession,
    newt_id: str,
    state: FSMContext,
):
    data = await state.get_data()

    belly_bytes = data["belly_bytes"]
    embedding = np.array(data["embedding"], dtype=np.float32)

    await insert_new_belly(
        session=session,
        image_bytes=belly_bytes,
        embedding=embedding,
        newt_class_name=newt_id,
        run_name="current_model",
    )
