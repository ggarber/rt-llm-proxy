import argparse
import asyncio
import fractions
import logging
import re
import ssl
import uuid

from aiohttp import web
from aiortc import (
    MediaStreamTrack,
    RTCPeerConnection,
    RTCSessionDescription,
    RTCConfiguration,
)

from av import AudioFrame
from PIL import Image

from model_gemini import connect_gemini
from model_openai import connect_openai

AUDIO_PTIME = 0.02
AUDIO_BITRATE = 16000
USE_VIDEO_BUFFER = False

logger = logging.getLogger("proxy")
connections = set()


class SendingTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue = asyncio.Queue()

    async def recv(self):
        return await self.queue.get()


class RTCConnection:
    recv_audio_track = None
    recv_video_track = None
    send_track = None
    pc = None
    genai_session = None

    async def handle_offer(self, request):
        content = await request.text()
        offer = RTCSessionDescription(sdp=content, type="offer")

        self.pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))

        model = request.query.get("model")
        asyncio.ensure_future(self._run(model))

        await self.pc.setRemoteDescription(offer)

        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)

        sdp = self.pc.localDescription.sdp
        found = re.findall(r"a=rtpmap:(\d+) opus/48000/2", sdp)
        if found:
            sdp = sdp.replace(
                "opus/48000/2\r\n",
                "opus/48000/2\r\n"
                + f"a=fmtp:{found[0]} useinbandfec=1;usedtx=1;maxaveragebitrate={AUDIO_BITRATE}\r\n",
            )

        return web.Response(
            content_type="application/sdp",
            text=sdp,
        )

    async def _run(self, model):
        pc_id = str(uuid.uuid4())

        def log_info(msg, *args):
            logger.info(pc_id + " " + msg, *args)

        log_info("Connection started")

        @self.pc.on("datachannel")
        def on_datachannel(channel):
            @channel.on("message")
            async def on_message(message):
                if self.genai_session:
                    await self.genai_session.send(message)

        @self.pc.on("connectionstatechange")
        async def on_connectionstatechange():
            log_info("Connection state is %s", self.pc.connectionState)
            if (
                self.pc.connectionState == "failed"
                or self.pc.connectionState == "closed"
            ):
                await self.close()

        @self.pc.on("track")
        def on_track(track):
            log_info("Track %s received", track.kind)

            if track.kind == "audio":
                # Only accept the first track received for now
                if self.recv_audio_track:
                    return

                self.recv_audio_track = track
                self.send_track = SendingTrack()
                self.pc.addTrack(self.send_track)
                asyncio.ensure_future(run_recv_audio_track())

            elif track.kind == "video":
                # Only accept the first track received for now
                if self.recv_video_track:
                    return

                self.recv_video_track = track
                asyncio.ensure_future(run_recv_video_track())

            @track.on("ended")
            async def on_ended():
                log_info("Track %s ended", track.kind)

        async def run_recv_audio_track():
            while True:
                try:
                    frame = await self.recv_audio_track.recv()
                    if not self.genai_session:
                        continue
                    await self.genai_session.send(frame)

                except Exception as e:
                    log_info("Error receiving frame: %s", e)
                    break

        async def run_recv_video_track():
            buffer = []
            while self.pc and self.pc.connectionState != "closed":
                try:
                    frame = await self.recv_video_track.recv()
                    if not self.genai_session:
                        continue

                    image = frame.to_image()

                    if USE_VIDEO_BUFFER:
                        buffer.append(image)
                        if len(buffer) > 10:
                            buffer.pop(0)

                        # Compose horizontally all the images in buffer
                        composite = Image.new(
                            "RGB", (image.width * len(buffer), image.height)
                        )
                        for i in range(len(buffer)):
                            composite.paste(buffer[i], (image.width * i, 0))

                        await self.genai_session.send(composite)
                    else:
                        await self.genai_session.send(image)

                except Exception as e:
                    log_info("Error receiving frame: %s", e)
                    break

        async def run_send_track():
            timestamp = 0
            buffer = b""
            while self.pc and self.pc.connectionState != "closed":
                async for frame in self.genai_session.recv():
                    sample_rate = frame.sample_rate
                    samples = int(sample_rate * AUDIO_PTIME)
                    buffer += frame.to_ndarray().tobytes()

                    while len(buffer) / 2 >= samples:
                        frame = AudioFrame(format="s16", layout="mono", samples=samples)
                        frame.sample_rate = sample_rate
                        frame.planes[0].update(buffer[: samples * 2])
                        buffer = buffer[samples * 2 :]

                        timestamp += sample_rate * AUDIO_PTIME
                        frame.pts = timestamp
                        frame.time_base = fractions.Fraction(1, sample_rate)
                        await self.send_track.queue.put(frame)
                        await asyncio.sleep(AUDIO_PTIME)

        try:
            connect_genai = connect_openai if model == "openai" else connect_gemini
            async with connect_genai() as session:
                log_info("Connected to GenAI session")
                self.genai_session = session

                await run_send_track()
                log_info("Connection finished")

        except Exception as e:
            log_info("Error sending frame: %s", e)

        try:
            await self.close()
        except Exception as e:
            log_info("Error closing connection: %s", e)
        connections.discard(self)
        log_info(f"Connection stopped. Connections {len(connections)}")

    async def close(self):
        if self.pc:
            await self.pc.close()
            self.pc = None
        if self.genai_session:
            await self.genai_session.close()
            self.genai_session = None


async def offer(request):
    connection = RTCConnection()
    connections.add(connection)
    return await connection.handle_offer(request)


async def on_shutdown(app):
    # close peer connections
    coros = [conn.close() for conn in connections]
    await asyncio.gather(*coros)
    connections.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real Time LLM Proxy")
    parser.add_argument("--cert-file")
    parser.add_argument("--key-file")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("aioice").setLevel(level=logging.WARN)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_post("/", offer)
    app.router.add_static("/demo", "demo")
    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )
