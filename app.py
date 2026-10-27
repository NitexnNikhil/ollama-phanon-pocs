import asyncio
import os
import queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional

import requests
import streamlit as st
from aiohttp import web
from dotenv import load_dotenv
from google import genai
from google.genai import types


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite",
)

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://127.0.0.1:11434/v1/audio/speech",
)

PHONON_MODEL = os.getenv(
    "PHONON_MODEL",
    "phonon",
)

AUDIO_HOST = os.getenv(
    "AUDIO_HOST",
    "127.0.0.1",
)

AUDIO_PORT = int(
    os.getenv(
        "AUDIO_PORT",
        "8765",
    )
)

PHONON_SAMPLE_RATE = int(
    os.getenv(
        "PHONON_SAMPLE_RATE",
        "24000",
    )
)

PHONON_CHANNELS = int(
    os.getenv(
        "PHONON_CHANNELS",
        "1",
    )
)

MAX_TTS_WORKERS = int(
    os.getenv(
        "MAX_TTS_WORKERS",
        "2",
    )
)

MAX_PENDING_TTS = int(
    os.getenv(
        "MAX_PENDING_TTS",
        "4",
    )
)

TTS_BATCH_MAX_CHARS = int(
    os.getenv(
        "TTS_BATCH_MAX_CHARS",
        "120",
    )
)

TTS_MIN_CHARS = int(
    os.getenv(
        "TTS_MIN_CHARS",
        "35",
    )
)


# ============================================================
# VALIDATION
# ============================================================

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY is missing in .env")
    st.stop()


# ============================================================
# LOGGING
# ============================================================

def debug(message: str) -> None:
    print(
        f"[DEBUG {time.strftime('%H:%M:%S')}] "
        f"{message}",
        flush=True,
    )


# ============================================================
# GEMINI
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY,
)


# ============================================================
# SYSTEM INSTRUCTION
# ============================================================

SYSTEM_INSTRUCTION = """
You are a helpful website voice assistant.

Answer naturally and conversationally.

Important:
- Give the most useful information first.
- Keep normal answers concise.
- If the user asks for detail, provide a detailed answer.
- Use short natural sentences because your answer will be spoken aloud.
- Avoid unnecessary Markdown.
- Avoid tables unless absolutely necessary.
- Avoid excessive bullet points.
- Do not repeat the user's question.
- Do not use emojis excessively.
- Do not mention that your response will be converted to speech.
"""


# ============================================================
# AUDIO SERVER
# ============================================================

class AudioServer:
    """
    HTTP server + WebSocket audio server.

    Browser:
        GET  /
        WS   /ws/<client_id>

    Python:
        send_pcm(client_id, pcm_bytes)
    """

    def __init__(
        self,
        host: str,
        port: int,
        html_path: Path,
    ):
        self.host = host
        self.port = port
        self.html_path = html_path

        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None

        self.ready_event = threading.Event()

        self.clients: Dict[
            str,
            web.WebSocketResponse,
        ] = {}

        self.clients_lock = threading.Lock()

    # --------------------------------------------------------
    # START
    # --------------------------------------------------------

    def start(self) -> None:

        if (
            self.thread
            and self.thread.is_alive()
        ):
            return

        self.thread = threading.Thread(
            target=self._run,
            name="phonon-audio-server",
            daemon=True,
        )

        self.thread.start()

        if not self.ready_event.wait(
            timeout=10
        ):
            raise RuntimeError(
                "Audio server failed to start."
            )

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    def _run(self) -> None:

        self.loop = asyncio.new_event_loop()

        asyncio.set_event_loop(
            self.loop
        )

        app = web.Application()

        app.router.add_get(
            "/",
            self._index,
        )

        app.router.add_get(
            "/ws/{client_id}",
            self._websocket,
        )

        debug(
            "Starting WebSocket audio server "
            f"on {self.host}:{self.port}"
        )

        runner = web.AppRunner(
            app,
            access_log=None,
        )

        self.loop.run_until_complete(
            runner.setup()
        )

        site = web.TCPSite(
            runner,
            self.host,
            self.port,
        )

        self.loop.run_until_complete(
            site.start()
        )

        self.ready_event.set()

        debug(
            "WebSocket audio server started"
        )

        try:

            self.loop.run_forever()

        finally:

            self.loop.run_until_complete(
                runner.cleanup()
            )

            self.loop.close()

    # --------------------------------------------------------
    # INDEX
    # --------------------------------------------------------

    async def _index(
        self,
        request: web.Request,
    ):

        try:

            html = self.html_path.read_text(
                encoding="utf-8"
            )

        except Exception as exc:

            return web.Response(
                status=500,
                text=(
                    "Could not load "
                    f"voice_component.html: {exc}"
                ),
            )

        return web.Response(
            text=html,
            content_type="text/html",
        )

    # --------------------------------------------------------
    # WEBSOCKET
    # --------------------------------------------------------

    async def _websocket(
        self,
        request: web.Request,
    ):

        client_id = request.match_info[
            "client_id"
        ]

        ws = web.WebSocketResponse(
            heartbeat=30,
        )

        await ws.prepare(request)

        with self.clients_lock:

            old_client = self.clients.get(
                client_id
            )

            self.clients[
                client_id
            ] = ws

        if old_client is not None:

            try:
                await old_client.close()

            except Exception:
                pass

        debug(
            "Audio client connected: "
            f"{client_id}"
        )

        try:

            async for message in ws:

                if message.type == web.WSMsgType.TEXT:

                    if message.data == "ping":

                        await ws.send_str(
                            "pong"
                        )

                elif message.type == web.WSMsgType.ERROR:

                    debug(
                        "WebSocket error: "
                        f"{ws.exception()}"
                    )

        finally:

            with self.clients_lock:

                current = self.clients.get(
                    client_id
                )

                if current is ws:

                    del self.clients[
                        client_id
                    ]

            debug(
                "Audio client disconnected: "
                f"{client_id}"
            )

        return ws

    # --------------------------------------------------------
    # SEND PCM
    # --------------------------------------------------------

    async def _send_pcm(
        self,
        client_id: str,
        pcm_data: bytes,
    ):

        with self.clients_lock:

            ws = self.clients.get(
                client_id
            )

        if ws is None:

            debug(
                "No browser client for "
                f"{client_id}; dropping "
                f"{len(pcm_data)} bytes"
            )

            return

        if ws.closed:

            return

        try:

            await ws.send_bytes(
                pcm_data
            )

        except Exception as exc:

            debug(
                "PCM send failed: "
                f"{exc}"
            )

    def send_pcm(
        self,
        client_id: str,
        pcm_data: bytes,
    ) -> None:

        if not self.loop:

            debug(
                "Audio loop unavailable"
            )

            return

        future = asyncio.run_coroutine_threadsafe(
            self._send_pcm(
                client_id,
                pcm_data,
            ),
            self.loop,
        )

        try:

            future.result(
                timeout=5
            )

        except Exception as exc:

            debug(
                "send_pcm error: "
                f"{exc}"
            )

    # --------------------------------------------------------
    # SEND CONTROL MESSAGE
    # --------------------------------------------------------

    async def _send_text(
        self,
        client_id: str,
        text: str,
    ):

        with self.clients_lock:

            ws = self.clients.get(
                client_id
            )

        if ws is None:
            return

        if ws.closed:
            return

        try:

            await ws.send_str(
                text
            )

        except Exception as exc:

            debug(
                "Control message failed: "
                f"{exc}"
            )

    def send_text(
        self,
        client_id: str,
        text: str,
    ) -> None:

        if not self.loop:
            return

        future = asyncio.run_coroutine_threadsafe(
            self._send_text(
                client_id,
                text,
            ),
            self.loop,
        )

        try:

            future.result(
                timeout=5
            )

        except Exception as exc:

            debug(
                "send_text error: "
                f"{exc}"
            )


# ============================================================
# START AUDIO SERVER ONCE
# ============================================================

@st.cache_resource
def get_audio_server() -> AudioServer:

    html_path = (
        Path(__file__).resolve().parent
        / "voice_component.html"
    )

    server = AudioServer(
        host=AUDIO_HOST,
        port=AUDIO_PORT,
        html_path=html_path,
    )

    server.start()

    return server


audio_server = get_audio_server()


# ============================================================
# PHONON
# ============================================================

def generate_phonon_audio(
    text: str,
) -> bytes:

    started = time.perf_counter()

    debug(
        "PHONON START | "
        f"chars={len(text)} | "
        f"text={text!r}"
    )

    payload = {
        "model": PHONON_MODEL,
        "voice": "default",
        "input": text,
        "response_format": "pcm",
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120,
        stream=True,
    )

    debug(
        "PHONON HTTP | "
        f"status={response.status_code} | "
        f"content_type="
        f"{response.headers.get('content-type')}"
    )

    response.raise_for_status()

    chunks = []

    total_bytes = 0

    for chunk in response.iter_content(
        chunk_size=4096
    ):

        if not chunk:
            continue

        chunks.append(chunk)

        total_bytes += len(chunk)

    audio = b"".join(
        chunks
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    debug(
        "PHONON COMPLETE | "
        f"chars={len(text)} | "
        f"bytes={len(audio)} | "
        f"time={elapsed:.2f}s"
    )

    return audio


# ============================================================
# TEXT BATCHING
# ============================================================

def split_long_piece(
    text: str,
    max_chars: int,
):

    text = text.strip()

    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    words = text.split()

    pieces = []

    current = ""

    for word in words:

        candidate = (
            word
            if not current
            else current + " " + word
        )

        if (
            len(candidate)
            <= max_chars
        ):

            current = candidate

        else:

            if current:
                pieces.append(
                    current.strip()
                )

            current = word

    if current:
        pieces.append(
            current.strip()
        )

    return pieces


def extract_tts_batches(
    buffer: str,
    force: bool = False,
):

    batches = []

    while buffer:

        if len(buffer) >= TTS_BATCH_MAX_CHARS:

            cut = buffer.rfind(
                " ",
                0,
                TTS_BATCH_MAX_CHARS,
            )

            if cut <= 0:

                cut = (
                    TTS_BATCH_MAX_CHARS
                )

            piece = buffer[
                :cut
            ].strip()

            buffer = buffer[
                cut:
            ].lstrip()

            if piece:

                batches.extend(
                    split_long_piece(
                        piece,
                        TTS_BATCH_MAX_CHARS,
                    )
                )

            continue

        # ----------------------------------------------------
        # Sentence boundary
        # ----------------------------------------------------

        boundary = -1

        for index, char in enumerate(
            buffer
        ):

            if char in ".!?":

                next_index = (
                    index + 1
                )

                if (
                    next_index
                    >= len(buffer)
                ):

                    boundary = (
                        index + 1
                    )

                    break

                next_char = (
                    buffer[
                        next_index
                    ]
                )

                if (
                    next_char.isspace()
                    or next_char in "\"'”’)"
                ):

                    boundary = (
                        index + 1
                    )

                    break

        if boundary != -1:

            piece = buffer[
                :boundary
            ].strip()

            buffer = buffer[
                boundary:
            ].lstrip()

            if (
                piece
                and (
                    len(piece)
                    >= TTS_MIN_CHARS
                    or force
                )
            ):

                batches.append(
                    piece
                )

            continue

        break

    # --------------------------------------------------------
    # Force remaining text
    # --------------------------------------------------------

    if force and buffer.strip():

        remaining = buffer.strip()

        batches.extend(
            split_long_piece(
                remaining,
                TTS_BATCH_MAX_CHARS,
            )
        )

        buffer = ""

    return (
        batches,
        buffer,
    )


# ============================================================
# AUDIO QUEUE
# ============================================================

class OrderedTTSProcessor:

    def __init__(
        self,
        client_id: str,
        audio_server: AudioServer,
    ):

        self.client_id = client_id
        self.audio_server = audio_server

        self.executor = (
            ThreadPoolExecutor(
                max_workers=MAX_TTS_WORKERS,
                thread_name_prefix="phonon",
            )
        )

        self.pending = []

        self.sequence = 0

        self.next_sequence = 0

        self.lock = threading.Lock()

    # --------------------------------------------------------
    # SUBMIT
    # --------------------------------------------------------

    def submit(
        self,
        text: str,
    ):

        text = text.strip()

        if not text:
            return

        sequence = self.sequence

        self.sequence += 1

        debug(
            "TTS SUBMIT | "
            f"seq={sequence} | "
            f"chars={len(text)}"
        )

        future = self.executor.submit(
            generate_phonon_audio,
            text,
        )

        self.pending.append(
            (
                sequence,
                text,
                future,
            )
        )

        # Prevent unlimited memory growth.
        self._apply_backpressure()

    # --------------------------------------------------------
    # BACKPRESSURE
    # --------------------------------------------------------

    def _apply_backpressure(self):

        while (
            len(self.pending)
            >= MAX_PENDING_TTS
        ):

            self.flush_one()

    # --------------------------------------------------------
    # FLUSH NEXT IN ORDER
    # --------------------------------------------------------

    def flush_one(self):

        target = None

        for item in self.pending:

            if (
                item[0]
                == self.next_sequence
            ):

                target = item

                break

        if target is None:
            return

        sequence, text, future = target

        try:

            audio = future.result()

        except Exception as exc:

            debug(
                "TTS FAILED | "
                f"seq={sequence} | "
                f"error={exc}"
            )

            self.pending.remove(
                target
            )

            self.next_sequence += 1

            return

        debug(
            "TTS READY | "
            f"seq={sequence} | "
            f"bytes={len(audio)}"
        )

        # ----------------------------------------------------
        # Send PCM in chunks.
        # ----------------------------------------------------

        PCM_SEND_SIZE = 8192

        for index in range(
            0,
            len(audio),
            PCM_SEND_SIZE,
        ):

            chunk = audio[
                index:
                index + PCM_SEND_SIZE
            ]

            self.audio_server.send_pcm(
                self.client_id,
                chunk,
            )

        self.pending.remove(
            target
        )

        self.next_sequence += 1

    # --------------------------------------------------------
    # FLUSH ALL
    # --------------------------------------------------------

    def flush_all(self):

        while self.pending:

            self.flush_one()

    # --------------------------------------------------------
    # CLOSE
    # --------------------------------------------------------

    def close(self):

        self.flush_all()

        self.executor.shutdown(
            wait=True
        )


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="AI Voice Assistant",
    page_icon="🤖",
    layout="centered",
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "Settings"
    )

    st.write(
        f"**Gemini:** "
        f"`{GEMINI_MODEL}`"
    )

    st.write(
        f"**TTS:** "
        f"`{PHONON_MODEL}`"
    )

    st.write(
        "**Gemini quota:**"
    )

    st.caption(
        "15 RPM • 250K TPM • 500 RPD"
    )

    st.write(
        "**Phonon:**"
    )

    st.caption(
        f"{PHONON_SAMPLE_RATE} Hz / "
        f"16-bit / mono"
    )

    if st.button(
        "🗑️ New Chat",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "voice_client_id" not in st.session_state:

    st.session_state.voice_client_id = (
        uuid.uuid4().hex
    )


voice_client_id = (
    st.session_state.voice_client_id
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "🤖 AI Voice Assistant"
)

st.caption(
    "Gemini 3.5 Flash-Lite + Phonon"
)


# ============================================================
# VOICE COMPONENT
# ============================================================

voice_url = (
    f"http://{AUDIO_HOST}:"
    f"{AUDIO_PORT}/"
    f"?client_id={voice_client_id}"
)

st.iframe(
    voice_url,
    height=85,
    width="stretch",
    tab_index=0,
    alt="Voice audio controller",
)


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# CHAT INPUT
# ============================================================

prompt = st.chat_input(
    "Ask me anything..."
)


if prompt:

    # --------------------------------------------------------
    # USER
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message(
        "user"
    ):

        st.markdown(
            prompt
        )

    # --------------------------------------------------------
    # ASSISTANT
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        response_placeholder = (
            st.empty()
        )

        full_response = ""

        tts_buffer = ""

        processor = OrderedTTSProcessor(
            client_id=voice_client_id,
            audio_server=audio_server,
        )

        request_started = (
            time.perf_counter()
        )

        first_token_time = None

        first_tts_submit_time = None

        try:

            # ------------------------------------------------
            # BUILD GEMINI HISTORY
            # ------------------------------------------------

            contents = []

            for message in (
                st.session_state.messages
            ):

                role = (
                    "user"
                    if message["role"]
                    == "user"
                    else "model"
                )

                contents.append(
                    types.Content(
                        role=role,
                        parts=[
                            types.Part(
                                text=message[
                                    "content"
                                ]
                            )
                        ],
                    )
                )

            debug(
                "STARTING GEMINI | "
                f"model={GEMINI_MODEL}"
            )

            debug(
                "Conversation messages="
                f"{len(contents)}"
            )

            # ------------------------------------------------
            # GEMINI STREAM
            # ------------------------------------------------

            response_stream = (
                client.models
                .generate_content_stream(
                    model=GEMINI_MODEL,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=(
                            SYSTEM_INSTRUCTION
                        ),
                        temperature=0.7,
                        max_output_tokens=4096,
                    ),
                )
            )

            debug(
                "GEMINI STREAM CREATED"
            )

            for chunk in response_stream:

                text = getattr(
                    chunk,
                    "text",
                    None,
                )

                if not text:

                    continue

                if first_token_time is None:

                    first_token_time = (
                        time.perf_counter()
                    )

                    debug(
                        "FIRST GEMINI TOKEN | "
                        f"{first_token_time - request_started:.2f}s"
                    )

                full_response += text

                tts_buffer += text

                response_placeholder.markdown(
                    full_response
                    + "▌"
                )

                # --------------------------------------------
                # EXTRACT TTS BATCHES
                # --------------------------------------------

                batches, tts_buffer = (
                    extract_tts_batches(
                        tts_buffer,
                        force=False,
                    )
                )

                for batch in batches:

                    if first_tts_submit_time is None:

                        first_tts_submit_time = (
                            time.perf_counter()
                        )

                        debug(
                            "FIRST TTS BATCH | "
                            f"{first_tts_submit_time - request_started:.2f}s"
                        )

                    processor.submit(
                        batch
                    )

            # ------------------------------------------------
            # FLUSH REMAINING TEXT
            # ------------------------------------------------

            remaining_batches, tts_buffer = (
                extract_tts_batches(
                    tts_buffer,
                    force=True,
                )
            )

            for batch in remaining_batches:

                processor.submit(
                    batch
                )

            # ------------------------------------------------
            # WAIT FOR ORDERED AUDIO
            # ------------------------------------------------

            debug(
                "GEMINI COMPLETE | "
                f"chars={len(full_response)}"
            )

            processor.flush_all()

            # ------------------------------------------------
            # FINAL UI
            # ------------------------------------------------

            response_placeholder.markdown(
                full_response
            )

            total_time = (
                time.perf_counter()
                - request_started
            )

            debug(
                "REQUEST COMPLETE | "
                f"time={total_time:.2f}s"
            )

            if first_token_time:

                debug(
                    "LLM TTFB | "
                    f"{first_token_time - request_started:.2f}s"
                )

            if first_tts_submit_time:

                debug(
                    "TTS START | "
                    f"{first_tts_submit_time - request_started:.2f}s"
                )

        except Exception as exc:

            debug(
                "LLM/TTS ERROR | "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            response_placeholder.error(
                f"Error: {exc}"
            )

        finally:

            try:

                processor.close()

            except Exception as exc:

                debug(
                    "Processor close error: "
                    f"{exc}"
                )

    # --------------------------------------------------------
    # SAVE ASSISTANT RESPONSE
    # --------------------------------------------------------

    if full_response.strip():

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_response,
            }
        )