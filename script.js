// Application Configuration and Model Thresholds
const VIDEO_WIDTH = 1280;
const VIDEO_HEIGHT = 720;
const MAX_HANDS = 2;
const MODEL_COMPLEXITY = 1;
const MIN_DETECTION_CONFIDENCE = 0.5;
const MIN_TRACKING_CONFIDENCE = 0.5;
const NUM_FEATURES = 84; // Expected input size for the ONNX model
const STABILITY_THRESHOLD = 12; // Frames required to confirm a sign
const RESET_THRESHOLD = 20; // Frames without hands before resetting output
const CONFIDENCE_THRESHOLD = 0.4;
const CONFIDENCE_MULTIPLIER = 1.4;
const MAX_VISUAL_CONFIDENCE = 0.99;
const LINE_WIDTH_CONNECTOR = 4;
const LINE_WIDTH_LANDMARK = 2;
const SPEECH_RATE = 1.0;
const SPEECH_PITCH = 1.0;

// DOM Elements
const videoElement = document.getElementById('input_video');
const canvasElement = document.getElementById('output_canvas');
const ctx = canvasElement.getContext('2d');
const predictionDisplay = document.getElementById('prediction-display');
const outputEn = document.getElementById('output-en');
const outputMs = document.getElementById('output-ms');
const startBtn = document.getElementById('start-btn');
const deviceSelect = document.getElementById('device-select');
const loadingSpinner = document.getElementById('loading-spinner');
const cameraPlaceholder = document.getElementById('camera-placeholder');
const confContainer = document.getElementById('confidence-bar-container');
const confValue = document.getElementById('conf-value');
const confFill = document.getElementById('conf-fill');

// Sign Language to Text Mappings
const LABELS_MAP = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "J", 
    10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R", 18: "S", 
    19: "T", 20: "U", 21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z_0", 26: "Z_1", 
    27: "How are you?", 28: "Waalaikumussalam", 29: "Hello", 30: "I'm fine", 
    31: "Excuse me", 32: "Sorry", 33: "Salam", 34: "Regards", 35: "You're welcome", 
    36: "Well", 37: "Come", 38: "Birthday", 39: "Goodbye", 40: "Night", 
    41: "Morning", 42: "Please (Welcome)", 43: "Thank you", 44: "Please (Help)"
};

const MALAY_MAPPING = {
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

// Grammar and Sequence Rules
const HIDDEN_SIGNS = new Set(["How are you?"]);
const MODIFIERS = { "J": "I", "Please (Welcome)": "Goodbye" }; // Requires a previous specific sign to activate
const ACTIVATORS = { // Merges two separate signs into a single phrase
    "Hello|Waalaikumussalam": "Assalamualaikum",
    "Well|Morning": "Good Morning",
    "Well|Night": "Good Night",
    "Well|Birthday": "Happy Birthday",
    "Well|Come": "Welcome",
    "How are you?|I'm fine": "How are you?"
};

// Application State
let onnxSession;
let inputName = "float_input";
let camera = null;
let isCameraRunning = false;
let previousPrediction = null;
let consecutiveFrames = 0;
let nothingConsecutiveFrames = 0;
let lastPrintedChar = null;
let lastValidEntry = null;
let wasLastHidden = false;
let currentSequenceBase = null;
let sequenceStep = -1;
let sentenceHistory = []; 
let lastTypeWasLetter = false;

function getLastEntry() {
    if (sentenceHistory.length === 0) return null;
    return sentenceHistory[sentenceHistory.length - 1].en;
}

function removeLastEntry() {
    if (sentenceHistory.length > 0) {
        sentenceHistory.pop();
        lastTypeWasLetter = sentenceHistory.length > 0 ? sentenceHistory[sentenceHistory.length - 1].isLetter : false;
    }
}

async function init() {
    await loadModel();
    await getCameras();
    renderBuffer();
}

async function loadModel() {
    try {
        loadingSpinner.classList.remove('hidden');
        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
        onnxSession = await ort.InferenceSession.create('./model.onnx');
        inputName = onnxSession.inputNames[0];
        loadingSpinner.classList.add('hidden');
    } catch (error) {
        console.error(error);
        alert("Error loading model.onnx");
    }
}

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
    } catch (error) {
        console.error("Error fetching cameras:", error);
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
        width: { ideal: VIDEO_WIDTH }, 
        height: { ideal: VIDEO_HEIGHT } 
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

// MediaPipe Setup
const hands = new Hands({locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`});
hands.setOptions({
    maxNumHands: MAX_HANDS, 
    modelComplexity: MODEL_COMPLEXITY, 
    minDetectionConfidence: MIN_DETECTION_CONFIDENCE, 
    minTrackingConfidence: MIN_TRACKING_CONFIDENCE
});
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
        let evaluatedHands = [];
        
        // Evaluate all hands to find the most prominent ones based on bounding box area
        for (let i = 0; i < results.multiHandLandmarks.length; i++) {
            const landmarks = results.multiHandLandmarks[i];
            let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
            
            for (let lm of landmarks) {
                if (lm.x < minX) minX = lm.x;
                if (lm.x > maxX) maxX = lm.x;
                if (lm.y < minY) minY = lm.y;
                if (lm.y > maxY) maxY = lm.y;
            }
            
            evaluatedHands.push({
                originalIndex: i, 
                landmarks: landmarks,
                area: (maxX - minX) * (maxY - minY),
                handedness: results.multiHandedness[i].label
            });
        }

        evaluatedHands.sort((a, b) => b.area - a.area);
        let topHands = evaluatedHands.slice(0, MAX_HANDS);
        topHands.sort((a, b) => a.originalIndex - b.originalIndex);

        // Calculate global boundaries to normalise the dataset
        let allX = [];
        let allY = [];
        for (let hand of topHands) {
            for (let lm of hand.landmarks) {
                allX.push(lm.x);
                allY.push(lm.y);
            }
        }
        let globalMinX = Math.min(...allX);
        let globalMaxX = Math.max(...allX);
        let globalMinY = Math.min(...allY);

        let isSingleLeftHand = (topHands.length === 1 && topHands[0].handedness === "Left");
        let normalizedFeatures = [];
        
        for (let hand of topHands) {
            drawConnectors(ctx, hand.landmarks, HAND_CONNECTIONS, {color: '#FFFFFF', lineWidth: LINE_WIDTH_CONNECTOR});
            drawLandmarks(ctx, hand.landmarks, {color: '#3F4EEF', lineWidth: LINE_WIDTH_LANDMARK});

            for (let lm of hand.landmarks) {
                let normalizedX;
                let normalizedY = lm.y - globalMinY;

                // Geometry normalisation: Left hand geometry matches mirrored training data.
                // Right hand (and two-handed signs) must be flipped to match.
                if (isSingleLeftHand) {
                    normalizedX = lm.x - globalMinX;
                } else {
                    normalizedX = globalMaxX - lm.x;
                }

                normalizedFeatures.push(normalizedX);
                normalizedFeatures.push(normalizedY);
            }
        }

        // Pad with zeros to ensure the model always receives exactly 84 features
        while (normalizedFeatures.length < NUM_FEATURES) {
            normalizedFeatures.push(0.0);
        }

        runInference(normalizedFeatures.slice(0, NUM_FEATURES));

    } else {
        handleNoHand();
    }
    ctx.restore();
}

async function runInference(features) {
    if (!onnxSession) return;
    try {
        const inputTensor = new ort.Tensor('float32', Float32Array.from(features), [1, NUM_FEATURES]);
        const feeds = {};
        feeds[inputName] = inputTensor;

        const results = await onnxSession.run(feeds);
        const labelIdx = Number(results[onnxSession.outputNames[0]].data[0]);
        
        let confidence = results[onnxSession.outputNames[1]] ? results[onnxSession.outputNames[1]].data[labelIdx] : 1.0;
        let visualConfidence = confidence;
        
        // Artificially boost the visual confidence bar if above the baseline threshold
        if (confidence > CONFIDENCE_THRESHOLD) {
            visualConfidence = Math.min(confidence * CONFIDENCE_MULTIPLIER, MAX_VISUAL_CONFIDENCE);
        }

        const label = LABELS_MAP[labelIdx];
        handleLogic(label, visualConfidence);

    } catch (error) {
        console.error(error);
    }
}

function handleLogic(predictedLabel, confidence) {
    nothingConsecutiveFrames = 0;
    updateConfidenceBar(confidence);

    let malayText = MALAY_MAPPING[predictedLabel] || predictedLabel;
    predictionDisplay.innerHTML = `<span>${predictedLabel}</span><span class="malay-text">${malayText}</span>`;

    if (predictedLabel === previousPrediction) {
        consecutiveFrames++;
    } else {
        consecutiveFrames = 0;
        previousPrediction = predictedLabel;
    }

    // Only process the sign if it has been held stably for the required number of frames
    if (consecutiveFrames === STABILITY_THRESHOLD) {
        
        // Handle multi-step dynamic signs (e.g., drawing a 'Z' in the air)
        if (predictedLabel.includes("_") && !HIDDEN_SIGNS.has(predictedLabel)) {
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

            // Prevent spamming the same character repeatedly
            if (predictedLabel !== lastPrintedChar) {
                
                if (HIDDEN_SIGNS.has(predictedLabel)) {
                    lastValidEntry = predictedLabel;
                    wasLastHidden = true;
                    lastPrintedChar = predictedLabel;
                } else {
                    let shouldPrint = true;
                    let isModifierReplace = false;

                    // Modifier logic: Check if the current sign needs to overwrite the previous one
                    if (MODIFIERS[predictedLabel]) {
                        let required = MODIFIERS[predictedLabel];
                        
                        if (required.length === 1) {
                            if (sentenceHistory.length > 0) {
                                let lastItem = sentenceHistory[sentenceHistory.length - 1];
                                if (lastItem.isLetter && lastItem.en.toUpperCase().endsWith(required)) {
                                    isModifierReplace = true;
                                } else {
                                    shouldPrint = false;
                                }
                            } else {
                                shouldPrint = false;
                            }
                        } else {
                            if (lastValidEntry === required) {
                                isModifierReplace = true;
                            } else {
                                shouldPrint = false;
                            }
                        }
                    }

                    if (shouldPrint) {
                        let comboKey = lastValidEntry + "|" + predictedLabel;
                        
                        // Activator logic: Merge two distinct signs into a single new phrase
                        if (ACTIVATORS[comboKey]) {
                            let newWord = ACTIVATORS[comboKey];
                            removeLastSign(); 
                            addToSentence(newWord);
                            lastValidEntry = newWord;
                            wasLastHidden = false;
                        } else if (isModifierReplace) {
                            removeLastSign(); 
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

function removeLastSign() {
    if (sentenceHistory.length === 0 || wasLastHidden) return;

    let lastIdx = sentenceHistory.length - 1;
    let item = sentenceHistory[lastIdx];

    // If removing from a spelled word, just pop the last letter rather than the whole block
    if (item.isLetter && item.en.length > 0) {
        item.en = item.en.slice(0, -1);
        item.ms = item.ms.slice(0, -1);
        
        if (item.en.length === 0) {
            sentenceHistory.pop();
            lastTypeWasLetter = sentenceHistory.length > 0 ? sentenceHistory[sentenceHistory.length - 1].isLetter : false;
        }
    } else {
        sentenceHistory.pop();
        lastTypeWasLetter = sentenceHistory.length > 0 ? sentenceHistory[sentenceHistory.length - 1].isLetter : false;
    }
    renderBuffer();
}

function handleNoHand() {
    nothingConsecutiveFrames++;
    if (nothingConsecutiveFrames > RESET_THRESHOLD) {
        lastPrintedChar = null;
        predictionDisplay.innerHTML = '<span class="placeholder-text">...</span>';
        updateConfidenceBar(0);
    }
}

function updateConfidenceBar(confidence) {
    const percentage = Math.round(confidence * 100);
    confValue.innerText = `${percentage}%`;
    confFill.style.width = `${percentage}%`;
}

function addToSentence(text) {
    const isLetter = (text.length === 1 && text.match(/[A-Z]/i));
    let enText = text;
    let msText = MALAY_MAPPING[text] || text; 

    // Group consecutive single letters together into a spelling block
    if (isLetter) {
        if (lastTypeWasLetter && sentenceHistory.length > 0) {
            let lastIdx = sentenceHistory.length - 1;
            sentenceHistory[lastIdx].en += enText.toLowerCase();
            sentenceHistory[lastIdx].ms += enText.toLowerCase(); 
        } else {
            sentenceHistory.push({ 
                en: enText.toUpperCase(), 
                ms: enText.toUpperCase(), 
                isLetter: true 
            });
        }
        lastTypeWasLetter = true;
    } else {
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

    if (item.isLetter && item.en.length > 1) {
        item.en = item.en.slice(0, -1);
        item.ms = item.ms.slice(0, -1);
    } else {
        sentenceHistory.pop();
        lastTypeWasLetter = sentenceHistory.length > 0 ? sentenceHistory[sentenceHistory.length - 1].isLetter : false;
    }
    renderBuffer();
}

function removeSpecificEntry(entry) {
    if (sentenceHistory.length === 0) return;
    
    let lastItem = sentenceHistory[sentenceHistory.length - 1];

    if (lastItem.en === entry || lastItem.en === entry.toUpperCase()) {
        sentenceHistory.pop();
        lastTypeWasLetter = sentenceHistory.length > 0 ? sentenceHistory[sentenceHistory.length - 1].isLetter : false;
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

    outputEn.innerText = sentenceHistory.map(item => item.en).join(" ");
    outputMs.innerText = sentenceHistory.map(item => item.ms).join(" ");
    
    outputEn.scrollTop = outputEn.scrollHeight;
    outputMs.scrollTop = outputMs.scrollHeight;
}

function speakText(languageCode) {
    if (sentenceHistory.length === 0) return;

    window.speechSynthesis.cancel();
    const textToSpeak = sentenceHistory.map(item => item[languageCode]).join(" ");
    const utterance = new SpeechSynthesisUtterance(textToSpeak);

    if (languageCode === 'en') {
        utterance.lang = 'en-GB'; 
    } else if (languageCode === 'ms') {
        utterance.lang = 'ms-MY';
    }

    utterance.rate = SPEECH_RATE; 
    utterance.pitch = SPEECH_PITCH;
    window.speechSynthesis.speak(utterance);
}

init();