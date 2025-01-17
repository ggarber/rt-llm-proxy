import contextlib
from typing import AsyncGenerator, AsyncIterator
from google import genai
from av import AudioFrame, AudioResampler
from PIL.Image import Image
import io

from model import Model, Input, Output


SAMPLE_RATE = 16000
AUDIO_PTIME = 0.02


class Gemini(Model):
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
            await self.session.send(input=input, end_of_turn=True)
        elif isinstance(input, AudioFrame):
            for frame in self.resampler.resample(input):
                blob = genai.types.BlobDict(
                    data=frame.to_ndarray().tobytes(),
                    mime_type=f"audio/pcm;rate={SAMPLE_RATE}",
                )
                await self.session.send(input=blob)
        elif isinstance(input, Image):
            array = io.BytesIO()
            input.save(array, format="JPEG")

            blob = genai.types.BlobDict(
                data=array.getvalue(),
                mime_type="image/jpeg",
            )
            await self.session.send(input=blob)

    async def recv(self) -> AsyncIterator[Output]:
        received = self.session.receive()
        async for event in received:
            if event.data is None:
                # log_info(f"Server Message - {response}")
                continue
            mime_type = event.server_content.model_turn.parts[0].inline_data.mime_type
            sample_rate = int(mime_type.split("rate=")[1])

            frame = AudioFrame(format="s16", layout="mono", samples=len(event.data) / 2)
            frame.sample_rate = sample_rate
            frame.planes[0].update(event.data)

            yield frame

    async def close(self):
        if self.session is None:
            return
        await self.session.close()
        self.session = None


client = genai.Client(http_options={"api_version": "v1alpha"})


@contextlib.asynccontextmanager
async def connect_gemini() -> AsyncGenerator[Gemini, None]:
    async with client.aio.live.connect(
        model="gemini-2.0-flash-exp",
        config={"generation_config": {"response_modalities": ["AUDIO"]}},
    ) as session:
        yield Gemini(session)
