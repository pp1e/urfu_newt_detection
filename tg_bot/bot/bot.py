from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession

from settings.loader import BOT_TOKEN
from settings.network_config import PROXY_URL

session = AiohttpSession(
    timeout=120,
    proxy=PROXY_URL,
)

bot = Bot(
    token=BOT_TOKEN,
    session=session,
)
dp = Dispatcher()
