from aiogram import Router, types
from aiogram.fsm.context import FSMContext

choose_model_router = Router()

@choose_model_router.callback_query(lambda c: c.data.startswith("choose_"))
async def choose_model_handler(callback: types.CallbackQuery, state: FSMContext):
    model_type = callback.data.replace("choose_", "")

    await state.update_data(model=model_type)

    await callback.message.answer(
        f"Отлично! Теперь пришли фото для анализа "
        f"{'тритона Карелина' if model_type=='karelina' else 'Ребристого тритона'}"
    )
    await callback.answer()
