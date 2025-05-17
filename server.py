import os, tempfile, json
from fastapi import FastAPI, File, UploadFile, Form
import whisper
from moviepy.editor import VideoFileClip
import requests

app = FastAPI()
model = whisper.load_model("base")  # или small

WORKER_URL = "https://video-clipper-chat.kott-png14.workers.dev"

@app.post("/highlights_advanced")
async def highlights_advanced(
    file: UploadFile = File(...),
    clips: int = Form(3),
    length: float = Form(15.0),
    skiprate: int = Form(10),
    keywords: str = Form("viral,funny,highlights")
):
    # Сохраняем видео
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.write(await file.read()); tmp.flush()

    # Транскрибируем
    audio = VideoFileClip(tmp.name).audio
    mp3 = tmp.name + ".mp3"; audio.write_audiofile(mp3)
    result = model.transcribe(mp3, language="en")  # whisper
    segments = result["segments"]  # list of dicts with start, end, text

    highlights = []
    for i in range(0, len(segments), skiprate):
        # Собираем текстовое окно
        window = segments[i:i+int(length)]
        prompt_text = "".join([seg["text"]+"\n" for seg in window])
        prompt = (
            "You are an assistant that decides if this conversation is worth "
            f"highlighting for keywords {keywords.split(',')}. "
            "If YES, respond `YES\\nSTART: X\\nEND: Y`. Else respond `NO`."
            "\n\nConversation:\n" + prompt_text
        )
        # Запрос к воркеру, который сам вызывает gpt-4o-mini
        resp = requests.post(
            f"{WORKER_URL}/chat",
            json={"type": "llm", "prompt": prompt}
        )
        data = resp.json()
        text = data.get("text", "")
        if text.startswith("YES"):
            # парсим X и Y
            lines = text.splitlines()
            X = int(lines[1].split(":")[1].strip()) - 1
            Y = int(lines[2].split(":")[1].strip()) - 1
            start = segments[i+X]["start"]
            end   = segments[i+Y]["end"]
            highlights.append({"start": start, "end": end})
    # чистим временные файлы
    os.remove(mp3); os.remove(tmp.name)
    return {"segments": highlights}
