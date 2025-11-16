from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

def start_over_kb():
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(
                text="🔙 Вернуться в начало",
                callback_data="start_over"
            )
        ]
    ])
