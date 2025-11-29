from aiogram import Router, types
from aiogram.fsm.context import FSMContext

from bot.belly_detection.detect_belly import detect_belly
from aiogram.types import BufferedInputFile
from bot.keyboards.start_over_kb import start_over_kb

router = Router()


@router.message(lambda m: m.photo or m.document)
async def handle_image(message: types.Message, state: FSMContext):
    data = await state.get_data()
    model_type = data.get("model")

    if not model_type:
        await message.answer("Сначала выбери модель через /start")
        return

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
        model_type=model_type,
    )

    await message.answer_photo(
        BufferedInputFile(detection_result.overlay, filename="overlay.jpg"),
        caption="Получившаяся маска",
        reply_markup=start_over_kb()
    )

    await message.answer_document(
        BufferedInputFile(detection_result.belly, filename="belly.png"),
        caption="Вырезанное брюшко",
        reply_markup=start_over_kb()
    )
