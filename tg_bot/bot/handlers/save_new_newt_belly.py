from aiogram import Router, types
from aiogram.fsm.context import FSMContext

from bot.service.save_embedding_from_state import save_embedding_from_state
from bot.state import AddNewtBellyFlow
from db.engine import AsyncSessionFactory

save_new_newt_belly_router = Router()

@save_new_newt_belly_router.message(AddNewtBellyFlow.waiting_for_new_newt_id)
async def save_new_newt_belly_handler(message: types.Message, state: FSMContext):
    newt_id = message.text.strip()

    async with AsyncSessionFactory() as session:
        await save_embedding_from_state(
            session=session,
            newt_id=newt_id,
            state=state,
        )

    await message.answer(f"Фото брюшка сохранено для тритона {newt_id}")
    await state.clear()
