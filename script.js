const videoElement = document.getElementById('input_video');
const canvasElement = document.getElementById('output_canvas');
const ctx = canvasElement.getContext('2d');
const predictionDisplay = document.getElementById('prediction-display');

// NEW: Grab separate outputs
const outputEn = document.getElementById('output-en');
const outputMs = document.getElementById('output-ms');

const startBtn = document.getElementById('start-btn');
const deviceSelect = document.getElementById('device-select');
const loadingSpinner = document.getElementById('loading-spinner');
const cameraPlaceholder = document.getElementById('camera-placeholder');

const confContainer = document.getElementById('confidence-bar-container');
const confValue = document.getElementById('conf-value');
const confFill = document.getElementById('conf-fill');

// --- 1. CONFIGURATION ---

const labels_map = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "J", 
    10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R", 18: "S", 
    19: "T", 20: "U", 21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z_0", 26: "Z_1", 
    27: "How are you?", 28: "Waalaikumussalam", 29: "Hello", 30: "I'm fine", 
    31: "Excuse me", 32: "Sorry", 33: "Salam", 34: "Regards", 35: "You're welcome", 
    36: "Well", 37: "Come", 38: "Birthday", 39: "Goodbye", 40: "Night", 
    41: "Morning", 42: "Please (Welcome)", 43: "Thank you", 44: "Please (Help)"
};

const malay_mapping = {
    "A": "A", "B": "B", "C": "C", "D": "D", "E": "E", "F": "F", "G": "G", "H": "H", "I": "I", "J": "J",
    "K": "K", "L": "L", "M": "M", "N": "N", "O": "O", "P": "P", "Q": "Q", "R": "R", "S": "S",
    "T": "T", "U": "U", "V": "V", "W": "W", "X": "X", "Y": "Y", "Z_0": "Z", "Z_1": "Z",
    "How are you?": "Apa khabar?",
    "Waalaikumussalam": "Waalaikumussalam",
    "Hello": "Helo",
    "I'm fine": "Khabar baik",
    "Excuse me": "Maaf",
    "Sorry": "Maaf",
    "Salam": "Salam",
    "Regards": "Salam",
    "You're welcome": "Sama-sama",
    "Well": "Selamat",
    "Come": "Datang",
    "Birthday": "Hari Jadi",
    "Goodbye": "Selamat Jalan",
    "Night": "Malam",
    "Morning": "Pagi",
    "Please (Welcome)": "Sila",
    "Thank you": "Terima Kasih",
    "Please (Help)": "Tolong",
    "Assalamualaikum": "Assalamualaikum",
    "Good Morning": "Selamat Pagi",
    "Good Night": "Selamat Malam",
    "Happy Birthday": "Selamat Hari Jadi",
    "Welcome": "Selamat Datang"
};

const hidden_signs = new Set(["How are you?"]);
const modifiers = { "J": "I", "Please (Welcome)": "Goodbye" };
const activators = {
    "Hello|Waalaikumussalam": "Assalamualaikum",
    "Well|Morning": "Good Morning",
    "Well|Night": "Good Night",
    "Well|Birthday": "Happy Birthday",
    "Well|Come": "Welcome",
    "How are you?|I'm fine": "How are you?"
};

let onnxSession;
let inputName = "float_input";
let camera = null;
let isCameraRunning = false;

// --- STATE MANAGEMENT ---
let previousPrediction = null;
let consecutiveFrames = 0;
let nothingConsecutiveFrames = 0;

const STABILITY_THRESHOLD = 12; 
const RESET_THRESHOLD = 20;

let lastPrintedChar = null;
let lastValidEntry = null;
let wasLastHidden = false;
let currentSequenceBase = null;
let sequenceStep = -1;

// History Array: [{ en: "Hello", ms: "Helo", isLetter: false }, ...]
let sentenceHistory = []; 
let lastTypeWasLetter = false;

// --- INITIALIZATION ---
async function init() {
    await loadModel();
    await getCameras();
    renderBuffer();
}

// --- LOAD MODEL ---
async function loadModel() {
    try {
        loadingSpinner.classList.remove('hidden');
        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
        onnxSession = await ort.InferenceSession.create('./model.onnx');
        inputName = onnxSession.inputNames[0];
        console.log("Model loaded");
        loadingSpinner.classList.add('hidden');
    } catch (e) {
        console.error(e);
        alert("Error loading model.onnx");
    }
}

// --- CAMERA HANDLING ---
async function getCameras() {
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(device => device.kind === 'videoinput');
        
        deviceSelect.innerHTML = '<option value="" disabled selected>SELECT DEVICE</option>';
        videoDevices.forEach(device => {
            const option = document.createElement('option');
            option.value = device.deviceId;
            option.text = device.label || `Camera ${deviceSelect.length}`;
            deviceSelect.appendChild(option);
        });
    } catch (e) {
        console.error("Error fetching cameras:", e);
    }
}

async function toggleCamera() {
    if (isCameraRunning) {
        stopCamera();
    } else {
        startCamera();
    }
}

function startCamera() {
    if (!onnxSession) {
        alert("Wait for model to load!");
        return;
    }

    loadingSpinner.classList.remove('hidden');
    cameraPlaceholder.classList.add('hidden');
    
    camera = new Camera(videoElement, {
        onFrame: async () => {
            await hands.send({image: videoElement});
        },
        width: { ideal: 1280 }, 
        height: { ideal: 720 } 
    });

    camera.start().then(() => {
        isCameraRunning = true;
        startBtn.innerText = "STOP";
        startBtn.classList.add('stop');
        canvasElement.classList.add('active');
        loadingSpinner.classList.add('hidden');
    });
}

function stopCamera() {
    if (camera) {
        window.location.reload(); 
    }
}

// --- PREDICTION LOOP ---
const hands = new Hands({locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`});
hands.setOptions({maxNumHands: 2, modelComplexity: 1, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5});
hands.onResults(onResults);

function onResults(results) {
    const videoWidth = results.image.width;
    const videoHeight = results.image.height;

    if (canvasElement.width !== videoWidth || canvasElement.height !== videoHeight) {
        canvasElement.width = videoWidth;
        canvasElement.height = videoHeight;
    }

    ctx.save();
    ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    ctx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        
        for (const landmarks of results.multiHandLandmarks) {
            drawConnectors(ctx, landmarks, HAND_CONNECTIONS, {color: '#FFFFFF', lineWidth: 4});
            drawLandmarks(ctx, landmarks, {color: '#FF4B4B', lineWidth: 2});
        }

        let dataAux = [];
        let x_ = [];
        let y_ = [];

        for (const landmarks of results.multiHandLandmarks) {
            for (let lm of landmarks) {
                x_.push(lm.x);
                y_.push(lm.y);
            }
            let minX = Math.min(...x_);
            let minY = Math.min(...y_);
            for (let lm of landmarks) {
                dataAux.push(lm.x - minX);
                dataAux.push(lm.y - minY);
            }
        }

        while (dataAux.length < 84) {
            dataAux.push(0.0);
        }

        runInference(dataAux);
    } else {
        handleNoHand();
    }
    ctx.restore();
}

async function runInference(features) {
    if (!onnxSession) return;
    try {
        const inputTensor = new ort.Tensor('float32', Float32Array.from(features), [1, 84]);
        const feeds = {};
        feeds[inputName] = inputTensor;

        const results = await onnxSession.run(feeds);
        const labelIdx = Number(results[onnxSession.outputNames[0]].data[0]);
        
        let confidence = 0.0;
        if(results[onnxSession.outputNames[1]]) {
            confidence = results[onnxSession.outputNames[1]].data[labelIdx];
        } else {
            confidence = 1.0; 
        }

        let visualConfidence = confidence;
        if (confidence > 0.4) {
            visualConfidence = Math.min(confidence * 1.4, 0.99);
        }

        const label = labels_map[labelIdx];
        handleLogic(label, visualConfidence);

    } catch (e) {
        console.error(e);
    }
}

// --- LOGIC ENGINE ---
function handleLogic(predictedLabel, confidence) {
    nothingConsecutiveFrames = 0;
    updateConfidenceBar(confidence);

    let malayText = malay_mapping[predictedLabel] || predictedLabel;
    
    predictionDisplay.innerHTML = `
        <span>${predictedLabel}</span>
        <span class="malay-text">${malayText}</span>
    `;

    if (predictedLabel === previousPrediction) {
        consecutiveFrames++;
    } else {
        consecutiveFrames = 0;
        previousPrediction = predictedLabel;
    }

    if (consecutiveFrames === STABILITY_THRESHOLD) {
        
        if (predictedLabel.includes("_") && !hidden_signs.has(predictedLabel)) {
            let parts = predictedLabel.split("_");
            let baseName = parts[0];
            let step = parseInt(parts[1]);

            if (step === 0) {
                currentSequenceBase = baseName;
                sequenceStep = 0;
            } else if (step === 1 && currentSequenceBase === baseName && sequenceStep === 0) {
                addToSentence(baseName);
                sequenceStep = -1;
                currentSequenceBase = null;
                lastPrintedChar = baseName;
                lastValidEntry = baseName;
                wasLastHidden = false;
            }
        } else {
            currentSequenceBase = null;
            sequenceStep = -1;

            if (predictedLabel !== lastPrintedChar) {
                if (hidden_signs.has(predictedLabel)) {
                    lastValidEntry = predictedLabel;
                    wasLastHidden = true;
                    lastPrintedChar = predictedLabel;
                } else {
                    let shouldPrint = true;
                    let isModifierReplace = false;

                    if (modifiers[predictedLabel]) {
                        if (lastValidEntry === modifiers[predictedLabel]) isModifierReplace = true;
                        else shouldPrint = false;
                    }

                    if (shouldPrint) {
                        let comboKey = lastValidEntry + "|" + predictedLabel;
                        if (activators[comboKey]) {
                            let newWord = activators[comboKey];
                            if (!wasLastHidden) {
                                removeSpecificEntry(lastValidEntry); 
                            }
                            addToSentence(newWord);
                            lastValidEntry = newWord;
                            wasLastHidden = false;
                        } else if (isModifierReplace) {
                            if (!wasLastHidden) {
                                removeSpecificEntry(lastValidEntry);
                            }
                            addToSentence(predictedLabel);
                            lastValidEntry = predictedLabel;
                            wasLastHidden = false;
                        } else {
                            addToSentence(predictedLabel);
                            lastValidEntry = predictedLabel;
                            wasLastHidden = false;
                        }
                        lastPrintedChar = predictedLabel;
                    }
                }
            }
        }
    }
}

function handleNoHand() {
    nothingConsecutiveFrames++;
    if (nothingConsecutiveFrames > RESET_THRESHOLD) {
        lastPrintedChar = null;
        predictionDisplay.innerHTML = '<span class="placeholder-text">...</span>';
        updateConfidenceBar(0);
    }
}

// --- UI HELPERS ---

function updateConfidenceBar(confidence) {
    const percentage = Math.round(confidence * 100);
    confValue.innerText = `${percentage}%`;
    confFill.style.width = `${percentage}%`;
}

// --- BILINGUAL SENTENCE LOGIC ---

function addToSentence(text) {
    const isLetter = (text.length === 1 && text.match(/[A-Z]/i));
    
    let enText = text;
    let msText = malay_mapping[text] || text; 

    if (isLetter) {
        if (lastTypeWasLetter && sentenceHistory.length > 0) {
            // MERGE (Spelling): Append letter to existing word
            let lastIdx = sentenceHistory.length - 1;
            sentenceHistory[lastIdx].en += enText.toLowerCase();
            sentenceHistory[lastIdx].ms += enText.toLowerCase(); 
        } else {
            // NEW LETTER BLOCK
            sentenceHistory.push({ 
                en: enText.toUpperCase(), 
                ms: enText.toUpperCase(), 
                isLetter: true 
            });
        }
        lastTypeWasLetter = true;
    } else {
        // NEW WORD/PHRASE
        sentenceHistory.push({ 
            en: enText, 
            ms: msText, 
            isLetter: false 
        });
        lastTypeWasLetter = false;
    }
    
    renderBuffer();
}

function handleBackspace() {
    if (sentenceHistory.length === 0) return;

    let lastIdx = sentenceHistory.length - 1;
    let item = sentenceHistory[lastIdx];

    // LOGIC: Delete entire word if phrase, single letter if spelled
    if (item.isLetter && item.en.length > 1) {
        item.en = item.en.slice(0, -1);
        item.ms = item.ms.slice(0, -1);
    } else {
        // Remove the whole object
        sentenceHistory.pop();
        
        // Reset state for concatenation
        if (sentenceHistory.length > 0) {
            lastTypeWasLetter = sentenceHistory[sentenceHistory.length - 1].isLetter;
        } else {
            lastTypeWasLetter = false;
        }
    }
    renderBuffer();
}

function removeSpecificEntry(entry) {
    if (sentenceHistory.length === 0) return;
    
    let lastItem = sentenceHistory[sentenceHistory.length - 1];

    if (lastItem.en === entry || lastItem.en === entry.toUpperCase()) {
        sentenceHistory.pop();
        if (sentenceHistory.length > 0) {
            lastTypeWasLetter = sentenceHistory[sentenceHistory.length - 1].isLetter;
        } else {
            lastTypeWasLetter = false;
        }
    }
    renderBuffer();
}

function clearSentence() {
    sentenceHistory = [];
    lastTypeWasLetter = false;
    lastValidEntry = null;
    renderBuffer();
}

function renderBuffer() {
    if (sentenceHistory.length === 0) {
        outputEn.innerHTML = '<span class="placeholder-text">...</span>';
        outputMs.innerHTML = '<span class="placeholder-text">...</span>';
        return;
    }

    const enLine = sentenceHistory.map(item => item.en).join(" ");
    const msLine = sentenceHistory.map(item => item.ms).join(" ");

    outputEn.innerText = enLine;
    outputMs.innerText = msLine;
    
    // Auto scroll both
    outputEn.scrollTop = outputEn.scrollHeight;
    outputMs.scrollTop = outputMs.scrollHeight;
}

// Run Init
init();