// get DOM elements
const dataChannelLog = document.getElementById('data-channel'),
      iceConnectionLog = document.getElementById('ice-connection-state');

var pc = null;
var dc = null;

function createPeerConnection() {
    var config = {
    };

    pc = new RTCPeerConnection(config);

    pc.addEventListener('iceconnectionstatechange', () => {
        iceConnectionLog.textContent += ' -> ' + pc.iceConnectionState;
    }, false);
    iceConnectionLog.textContent = pc.iceConnectionState;

    // connect audio / video
    pc.addEventListener('track', (evt) => {
        if (evt.track.kind == 'video')
            document.getElementById('video').srcObject = evt.streams[0];
        else
            document.getElementById('audio').srcObject = evt.streams[0];
    });

    return pc;
}

function enumerateInputDevices() {
    const populateSelect = (select, devices) => {
        let counter = 1;
        devices.forEach((device) => {
            const option = document.createElement('option');
            option.value = device.deviceId;
            option.text = device.label || ('Device #' + counter);
            select.appendChild(option);
            counter += 1;
        });
    };

    navigator.mediaDevices.enumerateDevices().then((devices) => {
        populateSelect(
            document.getElementById('audio-input'),
            devices.filter((device) => device.kind == 'audioinput')
        );
        populateSelect(
            document.getElementById('video-input'),
            devices.filter((device) => device.kind == 'videoinput')
        );
    }).catch((e) => {
        alert(e);
    });
}

async function negotiate() {
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    const response = await fetch('/', {
        body: offer.sdp,
        headers: {
            'Content-Type': 'application/sdp'
        },
        method: 'POST'
    });
 
    const sdp = await response.text();
    console.log(sdp)
    const answer = new RTCSessionDescription({
        type: 'answer',
        sdp: sdp
    });
    await pc.setRemoteDescription(answer);
}

function start() {
    document.getElementById('start').style.display = 'none';

    pc = createPeerConnection();

    dc = pc.createDataChannel('data', { ordered: true });
    dc.addEventListener('close', () => {
        dataChannelLog.textContent += '- close\n';
    });
    dc.addEventListener('open', () => {
        dataChannelLog.textContent += '- open\n';
    });
    dc.addEventListener('message', (evt) => {
        dataChannelLog.textContent += '< ' + evt.data + '\n';
    });

    // Build media constraints.

    const constraints = {
        audio: false,
        video: false,
    };

    if (document.getElementById('use-audio').checked) {
        const audioConstraints = {};

        const device = document.getElementById('audio-input').value;
        if (device) {
            audioConstraints.deviceId = { exact: device };
        }

        constraints.audio = Object.keys(audioConstraints).length ? audioConstraints : true;
    }

    if (document.getElementById('use-video').checked) {
        const videoConstraints = { frameRate: { max: 1 } };

        const device = document.getElementById('video-input').value;
        if (device) {
            videoConstraints.deviceId = { exact: device };
        }

        constraints.video = Object.keys(videoConstraints).length ? videoConstraints : true;
    }

    if (constraints.audio || constraints.video) {
        if (constraints.video) {
            document.getElementById('media').style.display = 'block';
        }
        navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
            stream.getTracks().forEach((track) => {
                pc.addTrack(track, stream);
            });
            return negotiate().catch(alert);
        }, (err) => {
            alert('Could not acquire media: ' + err);
        });
    } else {
        negotiate().catch(alert);
    }

    document.getElementById('stop').style.display = 'inline-block';
}

function stop() {
    document.getElementById('stop').style.display = 'none';

    if (pc) {
        pc.close();
        pc = null;
    }
}

enumerateInputDevices();

