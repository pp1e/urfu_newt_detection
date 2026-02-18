from aiogram import Router, types
from aiogram.fsm.context import FSMContext

from bot.database.is_newt_class_exist import is_newt_class_exist
from bot.service.save_embedding_from_state import save_embedding_from_state
from bot.state import AddNewtBellyFlow
from db.engine import AsyncSessionFactory

save_existing_newt_belly_router = Router()

@save_existing_newt_belly_router.message(AddNewtBellyFlow.waiting_for_existing_newt_id)
async def save_existing_newt_belly_handler(message: types.Message, state: FSMContext):
    newt_id = message.text.strip()

    async with AsyncSessionFactory() as session:
        is_exist = await is_newt_class_exist(
            session=session,
            newt_class=newt_id,
        )

        if not is_exist:
            await message.answer("Такой тритон не найден. Введите корректный ID:")
            return

    await save_embedding_from_state(
        session=session,
        newt_id=newt_id,
        state=state,
    )

    await message.answer(f"Фото брюшка сохранено для тритона {newt_id}")
    await state.clear()

