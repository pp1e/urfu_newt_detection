import asyncio
from bot.bot import bot, dp
from bot.handlers.choice_newt_handler import choice_newt_router
from bot.handlers.process_photo_handler import process_photo_router
from bot.handlers.save_existent_newt_belly_handler import save_existent_newt_belly_router
from bot.handlers.save_new_newt_belly_handler import save_new_newt_belly_router
from bot.handlers.start import start_router


async def main():
    dp.include_router(start_router)
    dp.include_router(process_photo_router)
    dp.include_router(choice_newt_router)
    dp.include_router(save_new_newt_belly_router)
    dp.include_router(save_existent_newt_belly_router)

    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
