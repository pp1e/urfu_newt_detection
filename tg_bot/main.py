import asyncio
from bot.bot import bot, dp
from bot.handlers.start import router as start_router
from bot.handlers.choose_model import router as choose_router
from bot.handlers.process_photo import router as photo_router

async def main():
    dp.include_router(start_router)
    dp.include_router(choose_router)
    dp.include_router(photo_router)

    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
