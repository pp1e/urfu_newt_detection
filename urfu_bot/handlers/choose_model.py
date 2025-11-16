from aiogram import Router, types
from aiogram.fsm.context import FSMContext

router = Router()

@router.callback_query(lambda c: c.data.startswith("choose_"))
async def choose_model(callback: types.CallbackQuery, state: FSMContext):
    model_type = callback.data.replace("choose_", "")

    await state.update_data(model=model_type)

    await callback.message.answer(
        f"Отлично! Теперь пришли фото для анализа ({'Карелина' if model_type=='karelina' else 'Ребристый'})"
    )
    await callback.answer()
