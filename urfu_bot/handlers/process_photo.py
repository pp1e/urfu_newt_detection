from aiogram import Router, types
from aiogram.fsm.context import FSMContext
from urfu_bot.inference import predict_mask
from aiogram.types import BufferedInputFile
from urfu_bot.keyboards.start_over_kb import start_over_kb

router = Router()


@router.message(lambda m: m.photo or m.document)
async def handle_image(message: types.Message, state: FSMContext):
    data = await state.get_data()
    model_type = data.get("model")

    if not model_type:
        await message.answer("Сначала выбери модель через /start")
        return

    # ================================
    # 1) Если отправлено КАК ФАЙЛ
    # ================================
    if message.document:
        doc = message.document

        # проверяем, что это изображение
        if not doc.mime_type.startswith("image/"):
            await message.answer("Пожалуйста, отправь изображение или фото.")
            return

        file = await message.bot.get_file(doc.file_id)
        photo_bytes = await message.bot.download_file(file.file_path)
        image_data = photo_bytes.read()

    # ================================
    # 2) Если отправлено КАК ФОТО
    # ================================
    else:
        file_id = message.photo[-1].file_id
        file = await message.bot.get_file(file_id)
        photo_bytes = await message.bot.download_file(file.file_path)
        image_data = photo_bytes.read()

    # ================================
    # 3) Инференс
    # ================================
    overlay_jpg, cutout_png = predict_mask(image_data, model_type)

    # Overlay
    await message.answer_photo(
        BufferedInputFile(overlay_jpg, filename="overlay.jpg"),
        caption="Вот результат сегментации 🦎",
        reply_markup=start_over_kb()
    )

    # vertical strip (вырезанное брюшко)
    await message.answer_document(
        BufferedInputFile(cutout_png, filename="belly.png"),
        caption="Вырезанное брюшко",
        reply_markup=start_over_kb()
    )
