import base64
import contextlib
from typing import AsyncGenerator, AsyncIterator
from openai import AsyncOpenAI
from av import AudioFrame, AudioResampler
from PIL.Image import Image
import io

from model import Model, Input, Output

SAMPLE_RATE = 24000
AUDIO_PTIME = 0.02


class OpenAI(Model):
    def __init__(self, session):
        self.session = session
        self.resampler = AudioResampler(
            format="s16",
            layout="mono",
            rate=SAMPLE_RATE,
            frame_size=int(SAMPLE_RATE * AUDIO_PTIME),
        )

    async def send(self, input: Input):
        if isinstance(input, str):
            await self.session.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Say hello!"}],
                }
            )
            await self.session.response.create()
        elif isinstance(input, AudioFrame):
            for frame in self.resampler.resample(input):
                data = frame.to_ndarray().tobytes()
                audio = base64.b64encode(data).decode("utf-8")
                await self.session.input_audio_buffer.append(audio=audio)
        elif isinstance(input, Image):
            array = io.BytesIO()
            input.save(array, format="JPEG")
            video = base64.b64encode(array.getvalue()).decode("utf-8")

            await self.session.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{video}"},
                        }
                    ],
                }
            )

    async def recv(self) -> AsyncIterator[Output]:
        async for event in self.session:
            if event.type == "response.audio.delta":
                data = base64.b64decode(event.delta)

                frame = AudioFrame(format="s16", layout="mono", samples=len(data) / 2)
                frame.sample_rate = SAMPLE_RATE
                frame.planes[0].update(data)

                yield frame

    async def close(self):
        await self.session.close()


client = AsyncOpenAI()


@contextlib.asynccontextmanager
async def connect_openai() -> AsyncGenerator[OpenAI, None]:
    async with client.beta.realtime.connect(
        model="gpt-4o-realtime-preview-2024-10-01"
    ) as conn:
        yield OpenAI(conn)
