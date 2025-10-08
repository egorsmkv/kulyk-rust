import requests

addr = "http://127.0.0.1:3000/translate"


def call(text: str):
    resp = requests.post(
        addr, json={"text": text, "source_lang": "en", "target_lang": "uk"}
    )
    print(resp.json())


def call_back(text: str):
    resp = requests.post(
        addr, json={"text": text, "source_lang": "uk", "target_lang": "en"}
    )
    print(resp.json())


if __name__ == "__main__":
    sentences_ukrainian = [
        "Привіт, світе!",
        "Як справи?",
        "Я вивчаю програмування на Rust.",
        "Це дуже цікаво.",
        "Дякую за допомогу.",
        "Гарного дня!",
        "До побачення.",
        "Слава Україні!",
        "Героям слава!",
        "Україна понад усе!",
    ]

    for sent in sentences_ukrainian:
        call(sent)

    sentences_english = [
        "Hello, world!",
        "How are you?",
        "I'm learning Rust programming.",
        "It's very interesting.",
        "Thank you for your help.",
        "Have a nice day!",
        "Goodbye.",
        "Glory to Ukraine!",
        "Glory to the heroes!",
        "Ukraine above all!",
    ]

    for sent in sentences_english:
        call_back(sent)
