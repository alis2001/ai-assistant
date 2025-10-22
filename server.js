import net from "net";
import fs from "fs";
import wav from "node-wav";
import { exec } from "child_process";

const PORT = 5050;
const FRAME_SIZE = 320;
const RECORDINGS_DIR = "./recordings";
const AUDIO_FILE = "./audio/intro.wav";

// Soglia minima dati IN attesi (in byte) — ad esempio 8kHz × 2 byte × durata_sec × canale 1
const MIN_IN_BYTES_FACTOR = 0.2; // se riceviamo meno del 20% del previsto => avviso

if (!fs.existsSync(RECORDINGS_DIR)) fs.mkdirSync(RECORDINGS_DIR);

/** Float32 → PCM16 */
function floatTo16BitPCM(float32Array) {
  const buffer = Buffer.alloc(float32Array.length * 2);
  for (let i = 0; i < float32Array.length; i++) {
    let s = Math.max(-1, Math.min(1, float32Array[i]));
    buffer.writeInt16LE(s < 0 ? s * 0x8000 : s * 0x7fff, i * 2);
  }
  return buffer;
}

/** Downsample audio → 8kHz */
function downsampleTo8kHz(input, inputRate) {
  if (inputRate === 8000) return input;
  const ratio = inputRate / 8000;
  const newLength = Math.round(input.length / ratio);
  const result = new Float32Array(newLength);
  let offsetResult = 0,
    offsetBuffer = 0;
  while (offsetResult < result.length) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio);
    let sum = 0,
      count = 0;
    for (let i = offsetBuffer; i < nextOffsetBuffer && i < input.length; i++) {
      sum += input[i];
      count++;
    }
    result[offsetResult++] = sum / count;
    offsetBuffer = nextOffsetBuffer;
  }
  return result;
}

/** Prepara PCM da file WAV */
async function preparePCM(wavPath) {
  const decoded = wav.decode(fs.readFileSync(wavPath));
  const mono = decoded.channelData[0];
  const downsampled = downsampleTo8kHz(mono, decoded.sampleRate);
  return floatTo16BitPCM(downsampled);
}

/** Decodifica G.711 A-law */
function alawToLinearSample(a_val) {
  a_val ^= 0x55;
  let t = (a_val & 0x0F) << 4;
  let seg = (a_val & 0x70) >> 4;
  switch (seg) {
    case 0: t += 8; break;
    case 1: t += 0x108; break;
    default: t += 0x108; t <<= seg - 1;
  }
  return (a_val & 0x80) ? t : -t;
}

function decodeAlaw(buffer) {
  const pcm = Buffer.alloc(buffer.length * 2);
  for (let i = 0; i < buffer.length; i++) {
    const sample = alawToLinearSample(buffer[i]);
    pcm.writeInt16LE(sample, i * 2);
  }
  return pcm;
}

/** Header frame AudioSocket */
const getHeader = () => Buffer.from([0x10, 0x01, 0x40]);
const silenceFrame = Buffer.alloc(FRAME_SIZE, 0);

/** Server TCP AudioSocket */
const server = net.createServer(async (socket) => {
  const callId = Date.now();
  console.log("📞 Nuova chiamata AudioSocket:", callId);

  const inFile = `${RECORDINGS_DIR}/call_${callId}_in.raw`;
  const outFile = `${RECORDINGS_DIR}/call_${callId}_out.raw`;
  const writeIn = fs.createWriteStream(inFile);
  const writeOut = fs.createWriteStream(outFile);

  // Prepara audio da inviare
  const pcm = await preparePCM(AUDIO_FILE);
  console.log(`🎧 Audio convertito: ${pcm.length} byte`);

  let offset = 0;
  let audioFinished = false;
  let totalFramesIn = 0;
  let totalBytesIn = 0;

  /** Invio audio OUT */
  const sendAudio = () => {
    if (offset < pcm.length) {
      let frame = pcm.slice(offset, offset + FRAME_SIZE);
      if (frame.length < FRAME_SIZE) {
        frame = Buffer.concat([frame, Buffer.alloc(FRAME_SIZE - frame.length)]);
      }
      socket.write(Buffer.concat([getHeader(), frame]));
      writeOut.write(frame);
      offset += FRAME_SIZE;
      setTimeout(sendAudio, 20);
    } else if (!audioFinished) {
      audioFinished = true;
      console.log("Playback completato. Invio silenzio per 10s...");
      const silenceTimer = setInterval(() => {
        socket.write(Buffer.concat([getHeader(), silenceFrame]));
        writeOut.write(silenceFrame);
      }, 20);

      setTimeout(() => {
        clearInterval(silenceTimer);
        console.log("Fine silenzio → interrompo invio. Asterisk chiuderà il bridge.");
        socket.end();
      }, 10000);
    }
  };

  setTimeout(sendAudio, 500);

  /** Ricezione audio IN */
  let bufferAcc = Buffer.alloc(0);
  socket.on("data", (chunk) => {
    console.log(`Pacchetto grezzo ricevuto: ${chunk.length} byte`);
    console.log(`Primo byte: 0x${chunk[0]?.toString(16)}`);
    bufferAcc = Buffer.concat([bufferAcc, chunk]);

    while (bufferAcc.length >= 3) {
      const type = bufferAcc.readUInt8(0);
      const length = bufferAcc.readUInt16BE(1);
      if (bufferAcc.length < 3 + length) {
        break;
      }
      const payload = bufferAcc.slice(3, 3 + length);
      bufferAcc = bufferAcc.slice(3 + length);

      if (type === 0x10) {
        totalFramesIn++;
        totalBytesIn += payload.length;
        const pcmDecoded = decodeAlaw(payload);
        writeIn.write(pcmDecoded);

        if (totalFramesIn % 50 === 0) {
          console.log(`Frame #${totalFramesIn} ricevuto (${payload.length} byte)`);
        }
      } else if (type === 0x00) {
        console.log("🔚 Tipo 0x00 terminazione ricevuta");
      } else {
        console.log(`Frame tipo 0x${type.toString(16)} ignorato`);
      }
    }
  });

  socket.on("close", () => {
    console.log("Connessione chiusa. Converto in MP3...");
    writeIn.end();
    writeOut.end();

    const sizeIn = fs.statSync(inFile).size;
    const sizeOut = fs.statSync(outFile).size;
    console.log(`Dimensione file IN: ${sizeIn} byte`);
    console.log(`Dimensione file OUT: ${sizeOut} byte`);

    const expectedInMin = sizeOut * MIN_IN_BYTES_FACTOR;
    if (sizeIn < expectedInMin) {
      console.warn(`Ricevuto pochi byte IN rispetto agli OUT: ${sizeIn} < ${expectedInMin}`);
    }

    const mp3File = `${RECORDINGS_DIR}/call_${callId}.mp3`;
    const cmd = `ffmpeg -y \
  -f s16le -ar 8000 -ac 1 -i ${inFile} \
  -f s16le -ar 8000 -ac 1 -i ${outFile} \
  -filter_complex "[0:a][1:a]amerge=inputs=2[a]; [a]highpass=f=200,lowpass=f=3400,dynaudnorm[a]" \
  -map "[a]" -ac 2 -ar 16000 -b:a 192k ${mp3File}`;

    exec(cmd, (err, stdout, stderr) => {
      if (err) {
        console.error("Errore conversione MP3:", err);
        console.error(stderr);
      } else {
        console.log(`MP3 creato correttamente: ${mp3File}`);
      }
    });
  });

  socket.on("error", (err) => console.error("Socket error:", err.code));
});

server.listen(PORT, "0.0.0.0", () =>
  console.log(`AudioSocket server in ascolto su porta ${PORT}`)
);
