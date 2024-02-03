import logging
import asyncio
import os
from aiogram import Bot, Dispatcher, types
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.filters import Command
from aiogram import F
from modules import run_yolo_detection_video, run_multi_yolo_detection
from PIL import Image
from aiogram.types import FSInputFile, InlineKeyboardButton, InlineKeyboardMarkup
import cv2

TOKEN = # Ваш токен

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
    await callback.message.edit_text("Отправьте изображение/видео для анализа")


@dp.message(F.photo)
async def download_photo(message: types.Message):
    username = message.from_user.username
    await bot.download(
        message.photo[-1],
        destination=f"Images/main{username}.jpg"
    )

    # flag = run_multi_yolo_detection('yolov8n.pt', "C:/Users/025/Documents/Bykovskij/OpenCV/Images/main.jpg")

    wait_result = await bot.send_message(chat_id=message.chat.id, text="Ваше фото обрабатывается. Это может занять некоторое время...")
    flag = run_multi_yolo_detection(['yolov8n.pt', 'runs/detect/Last/weights/best.pt'], f"Images/main{username}.jpg", username)

    capt = "Обнаружено оружие!" if flag else "Вот Ваше фото с обработкой от нейросети"
    # await message.answer_document(FSInputFile(path='Images/ResultImage.jpg'), caption=capt)
    await bot.send_photo(chat_id=message.chat.id, photo=FSInputFile(path=f'Images/ResultImage{username}.jpg'), caption=capt)
    await bot.delete_message(chat_id=message.chat.id, message_id=wait_result.message_id)
    # await message.answer_document(FSInputFile(path='Images/ResultImage.jpg'), caption=capt)

@dp.message(F.video)
async def download_photo(message: types.Message):
    username = message.from_user.username
    await bot.download(
        message.video,
        destination=f"Video/main{username}.mp4"
    )
    asyncio.sleep(7)
    wait_result = await bot.send_message(chat_id=message.chat.id, text="Ваше видео обрабатывается. Это может занять некоторое время...")
    flag = run_yolo_detection_video('yolov8n.pt', f"Video/main{username}.mp4", username)

    if flag:
        content = {
            "police": {
                "name": "Позвонить полицию"
            },
            "cover": {
                "name": "Вызвать подкрепление"
            },
            "nothing": {
                "name": "Игнорировать"
            }
        }
        keys = [
            [InlineKeyboardButton(
                text=content[mode]["name"],
                callback_data=mode + " mode"
            )] for mode in content
        ]
        keyboard = InlineKeyboardMarkup(inline_keyboard=keys)
        await bot.send_photo(chat_id=message.chat.id, photo=FSInputFile(path=f'Images/Person Detected{username}.jpg'), caption="Обнаружен нарушитель!")
        # await bot.send_message(chat_id=message.chat.id, text="Что делать в этой ситуации?", reply_markup=keyboard)
        await bot.delete_message(chat_id=message.chat.id, message_id=wait_result.message_id)
        # await message.answer_document(FSInputFile(path='Images/Person Detected.jpg'), caption="Обнаружен нарушитель!")
    else:
        await bot.send_video(chat_id=message.chat.id, video=FSInputFile(path=f'Video/ResultVideo{username}.mp4'), caption="Вот Ваше обработанное видео")

@dp.callback_query(F.data == 'police mode')
async def callPolice(callback: types.CallbackQuery):
    builder = InlineKeyboardBuilder()
    builder.button(text="Да", callback_data="yes")
    builder.button(text="Нет", callback_data="no")
    await callback.message.edit_text(
        text="Вы уверены?",
        reply_markup=builder.as_markup()
    )

@dp.callback_query(F.data == 'yes')
async def calling(callback: types.CallbackQuery):
    await callback.message.edit_text(
        text="Вызываю полицию на Ваш адрес..."
    )

@dp.message()
async def text(message: types.Message):
    await message.answer("Непредусмотренное сообщение или неверный API для регестрации.")

async def main():
    await dp.start_polling(bot)


if __name__ == '__main__':
    asyncio.run(main())
