import logging
import asyncio
import os
from aiogram import Bot, Dispatcher, types
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.filters import Command
from aiogram import F
from modules import run_yolo_detection_video, run_multi_yolo_detection
from PIL import Image
from aiogram.types import FSInputFile
import cv2

TOKEN = "Ваш Telegram API Token"

bot = Bot(token=TOKEN)
dp = Dispatcher()
logging.basicConfig(level=logging.INFO)


@dp.message(Command('start'))
async def process_start_command(message: types.Message):
    builder = InlineKeyboardBuilder()
    builder.button(text="Сотрудник", callback_data="сотрудник")
    builder.button(text="Тестировщик", callback_data="тестировщик")
    await message.answer("Здравствуйте! Кем Вы являетесь?", reply_markup=builder.as_markup())


@dp.callback_query(F.data == 'сотрудник')
async def security(callback: types.CallbackQuery):
    builder = InlineKeyboardBuilder()
    builder.button(text="Нет, я тестировщик", callback_data="тестировщик")
    await callback.message.edit_text("Пожалуйста, введите API охранного предприятия", reply_markup=builder.as_markup())

    ############################################
    # Тут часть кода для работы с сотрудниками #
    ############################################


@dp.callback_query(F.data == 'тестировщик')
async def tester(callback: types.CallbackQuery):
    await callback.message.edit_text("Пожалуйста, отправьте медиафайл (фото или видео)")


@dp.message(F.photo)
async def download_photo(message: types.Message):
    await bot.download(
        message.photo[-1],
        destination=f"Images/main.jpg"
    )

    # flag = run_multi_yolo_detection('yolov8n.pt', "C:/Users/025/Documents/Bykovskij/OpenCV/Images/main.jpg")

    flag = run_multi_yolo_detection(['yolov8n.pt', 'runs/detect/Last/weights/best.pt'], "Images/main.jpg")

    capt = "Обнаружено оружие!" if flag else "Вот Ваше фото с обработкой от нейросети"
    # await message.answer_document(FSInputFile(path='Images/ResultImage.jpg'), caption=capt)
    await message.answer_document(FSInputFile(path='Images/ResultImage.jpg'), caption=capt)

@dp.message(F.video)
async def download_photo(message: types.Message):
    await bot.download(
        message.video,
        destination=f"Video/main.mp4"
    )

    run_yolo_detection_video('yolov8n.pt', "Video/main.mp4")

    await message.answer_document(FSInputFile(path='Video/ResultVideo.mp4'), caption="Вот Ваше обработанное видео")

@dp.message()
async def text(message: types.Message):
    await message.answer("Непредусмотренное сообщение или неверный API для регестрации.")

async def main():
    await dp.start_polling(bot)


if __name__ == '__main__':
    asyncio.run(main())
