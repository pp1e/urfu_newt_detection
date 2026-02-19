from aiogram import Router, types
from aiogram.fsm.context import FSMContext

from bot.belly_detection.detect_belly import detect_belly
from aiogram.types import BufferedInputFile

from bot.belly_vectorization.embed_image_bytes import embed_image_bytes
from bot.database.search_newt_by_belly import find_best_match
from bot.handlers.utils import escape_md
from bot.keyboards.сonfirm_keyboard import confirm_kb
from bot.state import AddNewtBellyFlow
from db.engine import AsyncSessionFactory
from settings.embedder_loader import EMBEDDER_MODEL

process_photo_router = Router()


@process_photo_router.message(lambda m: m.photo or m.document)
async def process_photo_handler(message: types.Message, state: FSMContext):
    if message.document:
        doc = message.document

        if not doc.mime_type.startswith("image/"):
            await message.answer("Пожалуйста, отправь изображение или фото.")
            return

        file = await message.bot.get_file(doc.file_id)
        photo_bytes = await message.bot.download_file(file.file_path)
        image_data = photo_bytes.read()

    else:
        file_id = message.photo[-1].file_id
        file = await message.bot.get_file(file_id)
        photo_bytes = await message.bot.download_file(file.file_path)
        image_data = photo_bytes.read()

    detection_result = detect_belly(
        image_bytes=image_data,
    )

    belly_emb = embed_image_bytes(EMBEDDER_MODEL, detection_result.belly)

    async with AsyncSessionFactory() as session:
        matches = await find_best_match(
            session, belly_emb, top_k=5,
        )

        if not matches:
            await message.answer("База пустая — не с чем сравнивать.", parse_mode="Markdown")
            return

        best = matches[0]
        best_newt_class_name_markdown = escape_md(best.newt_class_name)
        if best.similarity >= 0.75:
            verdict = f"Похоже на тритона **{best_newt_class_name_markdown}** (сходство {best.similarity:.3f})."
            predicted_newt_id = best.newt_class_name
        else:
            verdict = (f"Похоже, это **новый тритон** (лучшее сходство {best.similarity:.3f}"
                       f" с тритоном **{best_newt_class_name_markdown}**).")
            predicted_newt_id = None

        await state.update_data(
            belly_bytes=detection_result.belly,
            embedding=belly_emb.tolist(),
            predicted_newt_id=predicted_newt_id,
        )
        await state.set_state(AddNewtBellyFlow.waiting_for_confirmation)

        await message.answer_photo(
            BufferedInputFile(detection_result.overlay, filename="overlay.jpg"),
            caption="Получившаяся маска",
        )

        await message.answer_document(
            BufferedInputFile(detection_result.belly, filename="belly.png"),
            caption="Вырезанное брюшко",
        )

        await message.answer(
            f"{verdict}\nПодтверждаете?",
            parse_mode="Markdown",
            reply_markup=confirm_kb(),
        )
