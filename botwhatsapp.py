import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from profanity_check import predict_prob
import speech_recognition as sr
import transformers
import requests
from telegram import Update
from telegram.ext import Updater, MessageHandler, Filters, CallbackContext

nltk.download('punkt')

conversations = [
    ("Hello", "Hello, how can I assist you?"),
    ("What are you doing?", "I'm here to answer your questions."),
]

tokenizer = Tokenizer()
tokenizer.fit_on_texts([pair[0] for pair in conversations])

max_sequence_length = 50
sequences = tokenizer.texts_to_sequences([pair[0] for pair in conversations])
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_sequence_length),
    LSTM(128),
    Dense(len(tokenizer.word_index) + 1, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(padded_sequences, tokenizer.texts_to_sequences([pair[1] for pair in conversations]), epochs=10)

banned_users = set()

def is_profanity_free(input_text):
    probabilities = predict_prob([input_text])
    return probabilities[0] < 0.5

def chatbot_response(input_text, user_id):
    if user_id in banned_users:
        return "You are banned from using the chatbot."
    
    input_sequence = tokenizer.texts_to_sequences([input_text])
    padded_input = pad_sequences(input_sequence, maxlen=max_sequence_length, padding='post')
    response_sequence = model.predict(padded_input)
    response = tokenizer.sequences_to_texts(response_sequence)[0]

    if is_profanity_free(response):
        model_gpt4 = transformers.GPT4LMModel.from_pretrained("openai/gpt-4")
        prompt = "Chatbot: " + response
        generated_text = model_gpt4.generate(prompt=prompt, max_length=100)
        decoded_text = tokenizer.decode(generated_text)
        return decoded_text
    else:
        return "I'm sorry, but I cannot respond to offensive content."

def search_images_in_google(query):
    search_url = "https://www.google.com/search"
    params = {
        "q": query + " image",
        "tbm": "isch"
    }
    response = requests.get(search_url, params=params)
    return response.text

def search_videos_in_youtube(query):
    search_url = "https://www.youtube.com/results"
    params = {
        "search_query": query
    }
    response = requests.get(search_url, params=params)
    return response.text

def search_stickers(query):
    return "Sticker: Sticker URL"

def perform_search_and_send_response(query, user_id):
    if "image" in query:
        search_result = search_images_in_google(query)
        response = "Image search results:\n"
    elif "video" in query:
        search_result = search_videos_in_youtube(query)
        response = "Video search results:\n"
    elif "sticker" in query:
        sticker_url = search_stickers(query)
        response = "Sticker: " + sticker_url
    else:
        response = chatbot_response(query, user_id)

    return response

def handle_message(update: Update, context: CallbackContext):
    user_id = str(update.message.from_user.id)
    message_text = update.message.text

    bot_response = perform_search_and_send_response(message_text, user_id)
    update.message.reply_text(bot_response)

def handle_audio(update: Update, context: CallbackContext):
    user_id = str(update.message.from_user.id)
    audio = update.message.voice.get_file()

    recognizer = sr.Recognizer()
    with sr.AudioFile(audio.download()) as source:
        try:
            audio_text = recognizer.recognize_google(source)
            bot_response = perform_search_and_send_response(audio_text, user_id)
            update.message.reply_text(bot_response)
        except sr.UnknownValueError:
            update.message.reply_text("Sorry, I couldn't understand the audio.")

def main():
    updater = Updater(token='6469779492:AAGLFOlQl1z9IiYivJUYtM3YLmxzb395myE', use_context=True)
    dispatcher = updater.dispatcher
    message_handler = MessageHandler(Filters.text & ~Filters.command, handle_message)
    audio_handler = MessageHandler(Filters.voice, handle_audio)

    dispatcher.add_handler(message_handler)
    dispatcher.add_handler(audio_handler)
    
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
    