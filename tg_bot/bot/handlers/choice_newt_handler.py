
from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery

from bot.service.save_embedding_from_state import save_embedding_from_state
from bot.state import AddNewtBellyFlow
from db.engine import AsyncSessionFactory

choice_newt_router = Router()

@choice_newt_router.callback_query(AddNewtBellyFlow.waiting_for_newt_class_choice, F.data.startswith("pick:"))
async def choice_newt_handler(
    callback: CallbackQuery,
    state: FSMContext
):
    newt_class_choice = callback.data.split(":", 1)[1]

    if newt_class_choice == "new":
        await state.set_state(AddNewtBellyFlow.waiting_for_new_newt_id)
        await callback.message.answer("Введите ID нового тритона:")
        await callback.answer()
        return
    if newt_class_choice == "manual":
        await state.set_state(AddNewtBellyFlow.waiting_for_existent_newt_id)
        await callback.message.answer("Введите ID существующего тритона:")
        await callback.answer()
        return

    async with AsyncSessionFactory() as session:
        await save_embedding_from_state(
            session=session,
            newt_id=newt_class_choice,
            state=state,
        )

    await callback.message.answer(f"Сохранил фото брюшка для тритона {newt_class_choice}")
    await state.clear()
    await callback.answer()
