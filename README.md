# LLM Real-Time Proxy Server 
This repository contains a Python server proof of concept that acts as a proxy for interacting with large language model (LLM) WebSocket APIs. Currently, it supports Gemini and facilitates real-time communication using WebRTC and WebTransport.

## Getting Started

### Prerequisites
- Python installed on your system
- Install requirements with `pip install -r requirements.txt`
- A valid `GOOGLE_API_KEY`

### Running the Server
1. Set the `GOOGLE_API_KEY` environment variable and run the server:
   ```bash
   GOOGLE_API_KEY=XXX python proxy.py
   ```

2. Open the test page in your browser and navigate to:
   ```
   http://localhost:8080/demo/index.html
   ```

3. Click **Start** and begin talking with the LLM

## Using it in your own page

```js
const stream = await navigator.getUserMedia({ audio: true })
const pc = new RTCPeerConnection();
pc.ontrack = e => audioElement.srcObject = e.streams[0];
pc.addTrack(stream.getTracks()[0]);
const offer = await pc.createOffer();
await pc.setLocalDescription(offer);
const resp = await fetch('http://localhost:8080/', { method: 'POST', body: offer.sdp })
await pc.setRemoteDescription({ type: 'answer', sdp: await resp.text() })
```

## Status and Plans

```
[X] WebRTC interface
[X] Audio support
[X] Basic datachannels support
[X] Gemini integration
[ ] WebTransport interface (WIP)
[ ] Video support
[ ] Implement HTTP real-time control interface 
```
