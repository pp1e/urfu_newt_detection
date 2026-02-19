import asyncio
from bot.bot import bot, dp
from bot.handlers.confirm_belly_handler import confirm_belly_router
from bot.handlers.process_photo import process_photo_router
from bot.handlers.save_existing_newt_belly import save_existing_newt_belly_router
from bot.handlers.save_new_newt_belly import save_new_newt_belly_router
from bot.handlers.start import start_router


async def main():
    dp.include_router(start_router)
    dp.include_router(process_photo_router)
    dp.include_router(confirm_belly_router)
    dp.include_router(save_existing_newt_belly_router)
    dp.include_router(save_new_newt_belly_router)

    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
