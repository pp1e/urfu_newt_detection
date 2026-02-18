from aiogram import Router, types
from aiogram.fsm.context import FSMContext

from bot.service.save_embedding_from_state import save_embedding_from_state
from bot.state import AddNewtBellyFlow
from db.engine import AsyncSessionFactory

confirm_belly_router = Router()


@confirm_belly_router.message(AddNewtBellyFlow.waiting_for_confirmation)
async def confirm_belly_handler(message: types.Message, state: FSMContext):
    answer = message.text.lower()
    data = await state.get_data()

    predicted_newt_id = data.get("predicted_newt_id")

    if answer == "да":

        if predicted_newt_id is not None:
            async with AsyncSessionFactory() as session:
                await save_embedding_from_state(
                    session=session,
                    newt_id=predicted_newt_id,
                    state=state,
                )
            await message.answer(f"Фото брюшка сохранено для тритона {predicted_newt_id}")
            await state.clear()
            return
        else:
            await message.answer("Введите ID нового тритона:")
            await state.set_state(AddNewtBellyFlow.waiting_for_new_newt_id)
            return

    else:

        if predicted_newt_id is not None:
            await message.answer("Введите ID нового тритона:")
            await state.set_state(AddNewtBellyFlow.waiting_for_new_newt_id)
        else:
            await message.answer("Введите ID существующего тритона:")
            await state.set_state(AddNewtBellyFlow.waiting_for_existing_newt_id)
