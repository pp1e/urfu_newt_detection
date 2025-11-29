from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

def choose_model_kb() -> InlineKeyboardMarkup:
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="Тритон Карелина", callback_data="choose_karelina"),
            InlineKeyboardButton(text="Ребристый тритон", callback_data="choose_ribbed"),
        ]
    ])
    return kb
