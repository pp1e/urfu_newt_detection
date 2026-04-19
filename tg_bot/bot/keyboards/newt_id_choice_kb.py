from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.utils.keyboard import InlineKeyboardBuilder

from bot.domain.newt_match import NewtMatch


def newt_id_choice_kb(
    matches: list[NewtMatch],
) -> InlineKeyboardMarkup:
    keyboard = InlineKeyboardBuilder()

    for match in matches:
        label = f"{match.class_name} ({match.similarity:.2f})"
        keyboard.add(
            InlineKeyboardButton(text=label, callback_data=f"pick:{match.class_name}")
        )

    keyboard.add(InlineKeyboardButton(text="✏️ Ввести ID тритона вручную", callback_data="pick:manual"))
    keyboard.add(InlineKeyboardButton(text="🆕 Это новый тритон", callback_data="pick:new"))
    keyboard.adjust(1)
    return keyboard.as_markup()
