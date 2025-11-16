from aiogram import Router, types
from aiogram.filters import Command
from urfu_bot.keyboards.choose_model_kb import choose_model_kb

router = Router()

async def send_start_message(message: types.Message):
    await message.answer(
        "Привет! Выбери тип тритона для анализа 🦎",
        reply_markup=choose_model_kb()
    )

@router.message(Command("start"))
async def start_cmd(message: types.Message):
    await send_start_message(message)


@router.callback_query(lambda c: c.data == "start_over")
async def start_over(callback: types.CallbackQuery):
    await send_start_message(callback.message)
    await callback.answer()
