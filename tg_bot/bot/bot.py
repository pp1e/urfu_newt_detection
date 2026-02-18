from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession

from settings.loader import BOT_TOKEN

session = AiohttpSession(timeout=120)

bot = Bot(
    token=BOT_TOKEN,
    session=session,
)
dp = Dispatcher()
