from aiogram import Router, types
from aiogram.fsm.context import FSMContext

from bot.belly_detection.detect_belly_pipeline import detect_belly
from aiogram.types import BufferedInputFile

from bot.belly_vectorization.embed_image_bytes import embed_image_bytes
from bot.database.search_newt_by_belly import find_best_match
from bot.keyboards.newt_id_choice_kb import newt_id_choice_kb
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


    await message.answer(
        f"Идёт обработка фото, пожалуйста, подождите...",
    )

    detection_result = detect_belly(
        image_bytes=image_data,
    )

    await message.answer_photo(
        BufferedInputFile(detection_result.overlay, filename="overlay.jpg"),
        caption="Получившаяся маска",
    )

    await message.answer_document(
        BufferedInputFile(detection_result.belly, filename="belly.png"),
        caption="Вырезанное брюшко",
    )

    belly_embedding = embed_image_bytes(EMBEDDER_MODEL, detection_result.belly)
    async with AsyncSessionFactory() as session:
        newt_matches = await find_best_match(
            session=session,
            query_emb=belly_embedding,
            top_k=5,
        )

    if not newt_matches:
        await state.set_state(AddNewtBellyFlow.waiting_for_new_newt_id)
        await message.answer("В базе нет тритонов.\nВведите ID нового триотона:", parse_mode="Markdown")
        return

    await state.update_data(
        belly_bytes=detection_result.belly,
        embedding=belly_embedding.tolist(),
        candidate_ids=[match.class_name for match in newt_matches],
    )
    await state.set_state(AddNewtBellyFlow.waiting_for_newt_class_choice)

    await message.answer(
        text="Выберите, какой это тритон из списка наиболее похожих:",
        reply_markup=newt_id_choice_kb(newt_matches),
    )

