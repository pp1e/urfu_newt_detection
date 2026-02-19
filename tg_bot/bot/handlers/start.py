from aiogram import Router, types
from aiogram.filters import Command

start_router = Router()

async def send_start_message(message: types.Message):
    await message.answer(
        "Привет! Отправь мне фото тритона для анализа",
    )

@start_router.message(Command("start"))
async def start_cmd(message: types.Message):
    await send_start_message(message)


@start_router.callback_query(lambda c: c.data == "start_over")
async def start_over(callback: types.CallbackQuery):
    await send_start_message(callback.message)
    await callback.answer()
