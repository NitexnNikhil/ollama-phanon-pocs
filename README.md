# AI Voice Assistant PoC

This project is a small Streamlit-based voice chatbot that combines:

- Gemini for streamed text generation
- Phonon via an Ollama-compatible speech endpoint for TTS
- A browser audio bridge that streams PCM audio back to the page

The result is a chat experience where responses can be read on screen and spoken aloud in the browser.

## Features

- Streaming Gemini responses in a chat UI
- Incremental text-to-speech generation while the model is still answering
- WebSocket-based PCM audio delivery to the browser
- Per-session chat history with a `New Chat` reset button
- Configurable model names, audio host/port, and TTS batching behavior through environment variables

## Project Layout

- `app.py` — Streamlit app, Gemini streaming, and TTS orchestration
- `voice_component.html` — browser-side audio controller and WebSocket client
- `test.wav` — sample audio asset used for local experimentation
- `pyproject.toml` / `uv.lock` — project metadata and locked dependencies
- `requirements.txt` — minimal pip dependencies

## Requirements

- Python 3.14 or newer
- A Gemini API key
- A reachable Ollama-style speech endpoint for Phonon

## Setup

1. Create and activate a virtual environment.
2. Install dependencies.

```bash
uv sync
```

If you prefer `pip`, install the project dependencies directly:

```bash
pip install aiohttp exa-py google-genai ollama openai python-dotenv requests streamlit
```

3. Create a `.env` file in the project root.

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.5-flash-lite

OLLAMA_URL=http://127.0.0.1:11434/v1/audio/speech
PHONON_MODEL=phonon

AUDIO_HOST=127.0.0.1
AUDIO_PORT=8765

PHONON_SAMPLE_RATE=24000
PHONON_CHANNELS=1
MAX_TTS_WORKERS=2
MAX_PENDING_TTS=4
TTS_BATCH_MAX_CHARS=120
TTS_MIN_CHARS=35
```

## Run

```bash
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal, then click **Enable Voice** inside the embedded voice controller.

## How It Works

1. The Streamlit app starts a small local `aiohttp` server for browser audio.
2. The embedded `voice_component.html` connects to that server through WebSocket.
3. User prompts are sent to Gemini with streaming enabled.
4. As Gemini emits text, the app batches sentences and sends them to Phonon for TTS.
5. Generated PCM audio is streamed back to the browser and played inline.

## Notes

- The app stops immediately if `GEMINI_API_KEY` is missing.
- Voice playback may require a user gesture in the browser, so the first click on **Enable Voice** is expected.
- If audio does not play, confirm that the TTS endpoint in `OLLAMA_URL` is reachable and that `AUDIO_PORT` is not blocked.

## License

No license has been specified yet.
