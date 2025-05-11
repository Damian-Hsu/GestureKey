// 全局變數
let labels = [];
let samples = {};           // { label: [landmarkVectors...] }
let model = null;
let prediction = null;
let gestureStart = null;
let currentGesture = null;
let recognizing = false;
let isTraining = false;
let lastLandmarks = null; // This will store processed landmarks for one or two hands
let lastRawLandmarks = null; // This will store raw landmarks from MediaPipe
let isRecording = false;
let currentRecordingLabel = null;
let recordingInterval = null;

// DOM元素
const video = document.getElementById('video');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const predictionOverlay = document.getElementById('prediction-overlay');
const cameraStatus = document.getElementById('camera-status');
const trainingStatus = document.getElementById('training-status');
const progressBar = document.getElementById('progress');
const epochInfo = document.getElementById('epoch-info');
const toast = document.getElementById('toast');

// 目標樣本數，用於進度條顯示
const TARGET_SAMPLES_PER_LABEL = 30;

// 啟動
window.addEventListener('DOMContentLoaded', init);

// 初始化攝影機
async function initCamera() {
  try {
    showToast('正在啟動相機...');
    const stream = await navigator.mediaDevices.getUserMedia({ 
      video: { 
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: 'user'
      } 
    });
    video.srcObject = stream;
    // Apply horizontal flip AND centering for the video element
    video.style.transform = 'translate(-50%, -50%) scaleX(-1)'; 
    
    return new Promise(resolve => {
      video.onloadedmetadata = () => {
        console.log(`視頻尺寸: ${video.videoWidth} x ${video.videoHeight}`);
        cameraStatus.textContent = '相機已就緒';
        setTimeout(() => {
          cameraStatus.style.opacity = '0';
        }, 1000);
        resolve();
      };
    });
  } catch (error) {
    console.error('無法啟動相機:', error);
    cameraStatus.textContent = '相機啟動失敗';
    cameraStatus.classList.add('error');
    throw error;
  }
}

// 初始化 MediaPipe Hands
const hands = new Hands({ 
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}` 
});
hands.setOptions({
  maxNumHands: 2, // Changed to 2 to detect both hands
  modelComplexity: 1,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});
hands.onResults(onHandsResults);

// 初始化
async function init() {
  try {
    //setupEventListeners();
    // 一進來就清空儲存
    localStorage.clear();
    setupEventListeners();
    await initCamera();
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    drawLoop();
    //loadFromLocalStorage();
    showToast('系統已準備就緒！');
  } catch (error) {
    console.error('初始化失敗:', error);
    showToast('初始化失敗，請重新整理頁面', 'error');
  }
}

// 設置所有事件監聽器
function setupEventListeners() {
  document.getElementById('add-label').addEventListener('click', addNewLabel);
  document.getElementById('train').addEventListener('click', trainModel);
  document.getElementById('save-model').addEventListener('click', saveModel);
  document.getElementById('load-model').addEventListener('change', loadModel);
  document.getElementById('start-recognize').addEventListener('click', toggleRecognition);
  document.getElementById('clear-output').addEventListener('click', () => {
    document.getElementById('text-output').value = '';
  });
  document.getElementById('copy-output').addEventListener('click', () => {
    const textarea = document.getElementById('text-output');
    textarea.select();
    document.execCommand('copy');
    showToast('文字已複製到剪貼簿');
  });
  document.getElementById('hand-selection').addEventListener('change', (event) => {
    const selection = event.target.value;
    let newMaxNumHands;
    if (selection === 'left' || selection === 'right') {
      newMaxNumHands = 1;
    } else { // 'all'
      newMaxNumHands = 2;
    }
    hands.setOptions({ maxNumHands: newMaxNumHands });
    showToast(`手部偵測已切換為: ${selection === 'all' ? '雙手' : (selection === 'left' ? '左手' : '右手')}, 最大偵測數量: ${newMaxNumHands}`);
    if (lastRawLandmarks) {
      processHandLandmarks(lastRawLandmarks); // Re-process with new setting
    }
  });
}

// 調整畫布尺寸
function resizeCanvas() {
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  canvas.style.width = video.offsetWidth + 'px';
  canvas.style.height = video.offsetHeight + 'px';
  if (predictionOverlay) {
    predictionOverlay.style.fontSize = (video.offsetHeight * 0.05) + 'px';
  }
}

// 每幀送入 MediaPipe
async function drawLoop() {
  try {
    await hands.send({ 
      image: video
    });
  } catch (error) {
    console.error('處理影像錯誤:', error);
  }
  requestAnimationFrame(drawLoop);
}

// 處理手部偵測結果
function onHandsResults(results) {
  lastRawLandmarks = results;
  processHandLandmarks(results);

  if (isRecording && !lastLandmarks) {
    showToast('無法偵測到符合設定的手，請將手放在鏡頭前', 'warning');
  }
}

// 處理和過濾手部數據的函數
function processHandLandmarks(results) {
  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const handSelection = document.getElementById('hand-selection').value;
  let processedLandmarks = null;
  const displayLandmarks = [];

  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    const leftHandLandmarks = [];
    const rightHandLandmarks = [];

    results.multiHandLandmarks.forEach((landmarks, i) => {
      
      // Flip x-coordinate for drawing on the non-flipped canvas, to match the mirrored video
      // 原本的 label 是基於「未鏡像」影像，畫面上要鏡像後才看起來正確
      const originalLabel = results.multiHandedness[i].label;
      const handedness = originalLabel === 'Left' ? 'Right' : 'Left';
      const flippedLandmarks = landmarks.map(point => ({
        x: 1 - point.x, 
        y: point.y,
        z: point.z
      }));

      if (handedness === 'Left') {
        leftHandLandmarks.push(...flippedLandmarks);
      } else if (handedness === 'Right') {
        rightHandLandmarks.push(...flippedLandmarks);
      }
    });

    let finalLeft = leftHandLandmarks.length > 0 ? leftHandLandmarks : null;
    let finalRight = rightHandLandmarks.length > 0 ? rightHandLandmarks : null;

    if (handSelection === 'left') {
      if (finalLeft) {
        processedLandmarks = { left: finalLeft, right: null };
        displayLandmarks.push(...finalLeft);
      } else {
        processedLandmarks = { left: null, right: null };
      }
      finalRight = null;
    } else if (handSelection === 'right') {
      if (finalRight) {
        processedLandmarks = { left: null, right: finalRight };
        displayLandmarks.push(...finalRight);
      } else {
        processedLandmarks = { left: null, right: null };
      }
      finalLeft = null;
    } else {
      processedLandmarks = { left: finalLeft, right: finalRight };
      if (finalLeft) displayLandmarks.push(...finalLeft);
      if (finalRight) displayLandmarks.push(...finalRight);
    }

    if (displayLandmarks.length > 0) {
      if (handSelection === 'all' && finalLeft && finalRight) {
        drawConnectors(ctx, finalLeft, HAND_CONNECTIONS, { color: '#00FF00', lineWidth: 3 });
        drawLandmarks(ctx, finalLeft, { color: '#FF0000', lineWidth: 1, radius: 3 });
        drawConnectors(ctx, finalRight, HAND_CONNECTIONS, { color: '#0000FF', lineWidth: 3 });
        drawLandmarks(ctx, finalRight, { color: '#FFFF00', lineWidth: 1, radius: 3 });
      } else {
        drawConnectors(ctx, displayLandmarks, HAND_CONNECTIONS, { color: '#00FF00', lineWidth: 3 });
        drawLandmarks(ctx, displayLandmarks, { color: '#FF0000', lineWidth: 1, radius: 3 });
      }
    }

    lastLandmarks = processedLandmarks;
    handleFrame(lastLandmarks);

    if (isRecording) {
      ctx.save();
      ctx.fillStyle = 'rgba(255, 0, 0, 0.5)';
      ctx.beginPath();
      ctx.arc(canvas.width / 2, canvas.height / 2, 20, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
  } else {
    lastLandmarks = null;
    if (recognizing) {
      predictionOverlay.textContent = '沒有偵測到手';
      predictionOverlay.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
    }
    resetGesture();
  }
}

// 將 21 個關鍵點 (單手或雙手) 轉成一維向量 (126個元素)
function landmarksToVector(handData) {
  const vector = [];
  const emptyHand = Array(21 * 3).fill(0);

  // 確保數值是有限的數字，否則返回 0
  const sanitize = (val) => (typeof val === 'number' && isFinite(val) ? val : 0);

  if (handData && handData.left) {
    vector.push(...handData.left.flatMap(pt => [sanitize(pt.x), sanitize(pt.y), sanitize(pt.z)]));
  } else {
    vector.push(...emptyHand);
  }

  if (handData && handData.right) {
    vector.push(...handData.right.flatMap(pt => [sanitize(pt.x), sanitize(pt.y), sanitize(pt.z)]));
  } else {
    vector.push(...emptyHand);
  }
  return vector;
}

// 新增標籤
function addNewLabel() {
  const input = document.getElementById('new-label');
  const name = input.value.trim();
  
  if (!name) {
    showToast('請輸入標籤名稱', 'error');
    return;
  }
  
  if (labels.includes(name)) {
    showToast('標籤已存在', 'error');
    return;
  }
  
  labels.push(name);
  samples[name] = [];
  renderLabels();
  input.value = '';
  showToast(`已新增「${name}」標籤`);
  saveToLocalStorage();
}

// 渲染標籤列表
function renderLabels() {
  const container = document.getElementById('labels');
  container.innerHTML = '';
  
  if (labels.length === 0) {
    container.innerHTML = '<div class="empty-state">尚未新增任何標籤</div>';
    return;
  }
  
  labels.forEach((label, index) => {
    const itemDiv = document.createElement('div');
    itemDiv.className = 'label-item';

    const labelInfoDiv = document.createElement('div');
    labelInfoDiv.className = 'label-info';
    
    const labelTextSpan = document.createElement('span');
    labelTextSpan.className = 'label-text';
    labelTextSpan.textContent = label;
    labelInfoDiv.appendChild(labelTextSpan);

    const sampleCountSpan = document.createElement('span');
    sampleCountSpan.className = 'sample-count-display';
    const currentSamples = samples[label] ? samples[label].length : 0;
    sampleCountSpan.textContent = `${currentSamples} 樣本`;
    labelInfoDiv.appendChild(sampleCountSpan);
    itemDiv.appendChild(labelInfoDiv);

    const progressBarDiv = document.createElement('div');
    progressBarDiv.className = 'label-progress-bar';
    const progressDiv = document.createElement('div');
    progressDiv.className = 'label-progress';
    const percentage = Math.min((currentSamples / TARGET_SAMPLES_PER_LABEL) * 100, 100);
    progressDiv.style.width = `${percentage}%`;
    progressBarDiv.appendChild(progressDiv);
    itemDiv.appendChild(progressBarDiv);

    const controlsDiv = document.createElement('div');
    controlsDiv.className = 'label-controls';

    const recordBtn = document.createElement('button');
    recordBtn.className = 'record-btn';
    recordBtn.innerHTML = '<span class="material-icons">fiber_manual_record</span> 錄製';
    recordBtn.title = '按一下拍一張，按住不放連續拍攝';
    recordBtn.addEventListener('mousedown', () => startRecordingByHold(label));
    recordBtn.addEventListener('mouseup', stopRecording);
    recordBtn.addEventListener('mouseleave', stopRecording);
    recordBtn.addEventListener('click', (e) => {
      e.preventDefault();
      if (!isRecording) {
        collectSingleSample(label);
      }
    });
    controlsDiv.appendChild(recordBtn);

    const clearSamplesBtn = document.createElement('button');
    clearSamplesBtn.className = 'clear-samples-btn';
    clearSamplesBtn.innerHTML = '<span class="material-icons">delete_sweep</span> 清空';
    clearSamplesBtn.title = '清空此標籤的所有樣本';
    clearSamplesBtn.onclick = () => clearSamples(label);
    controlsDiv.appendChild(clearSamplesBtn);
    
    const deleteBtn = document.createElement('button');
    deleteBtn.className = 'delete-label-btn';
    deleteBtn.innerHTML = '<span class="material-icons">delete</span> 刪除';
    deleteBtn.title = '刪除標籤和所有樣本';
    deleteBtn.onclick = () => deleteLabel(index);
    controlsDiv.appendChild(deleteBtn);
    
    itemDiv.appendChild(controlsDiv);
    container.appendChild(itemDiv);
  });
}

// 清空特定標籤的樣本
function clearSamples(label) {
  if (confirm(`確定要清空「${label}」標籤的所有樣本嗎？`)) {
    if (samples[label]) {
      samples[label] = [];
      renderLabels();
      showToast(`已清空「${label}」的樣本`);
      saveToLocalStorage();
    }
  }
}

// 切換錄製狀態
function toggleRecording(label) {
  if (isRecording && currentRecordingLabel === label) {
    stopRecording();
  } else {
    startRecording(label);
  }
}

// 開始錄製
function startRecording(label) {
  if (isRecording) {
    stopRecording();
  }
  
  if (!lastLandmarks) {
    showToast('無法偵測到符合設定的手，請將手放在鏡頭前開始錄製', 'warning');
    return;
  }
  
  isRecording = true;
  currentRecordingLabel = label;
  showToast(`正在錄製「${label}」樣本`);
  
  recordingInterval = setInterval(() => {
    if (lastLandmarks) {
      samples[label].push(landmarksToVector(lastLandmarks));
      renderLabels();
      showToast(`已收集「${label}」樣本: ${samples[label].length}`, 'info', 500);
    }
  }, 100);
  
  setTimeout(stopRecording, 5000);
}

// 停止錄製
function stopRecording() {
  if (!isRecording) return;
  
  clearInterval(recordingInterval);
  isRecording = false;
  if (currentRecordingLabel) {
    showToast(`已完成「${currentRecordingLabel}」樣本錄製`);
  }
  currentRecordingLabel = null;
  saveToLocalStorage();
}

// 收集單個樣本（點擊時）
function collectSingleSample(label) {
  if (!lastLandmarks) {
    showToast('無法偵測到符合設定的手，請將手放在鏡頭前', 'warning');
    return;
  }
  
  samples[label].push(landmarksToVector(lastLandmarks));
  renderLabels();
  showToast(`已收集「${label}」單個樣本: ${samples[label].length}`, 'success');
  saveToLocalStorage();
}

// 開始按住連續錄製
function startRecordingByHold(label) {
  if (isRecording) {
    stopRecording();
  }
  
  if (!lastLandmarks) {
    showToast('無法偵測到符合設定的手，請將手放在鏡頭前開始錄製', 'warning');
    return;
  }
  
  isRecording = true;
  currentRecordingLabel = label;
  showToast(`正在錄製「${label}」樣本`);
  
  recordingInterval = setInterval(() => {
    if (lastLandmarks) {
      samples[label].push(landmarksToVector(lastLandmarks));
      renderLabels();
      showToast(`已收集「${label}」樣本: ${samples[label].length}`, 'info', 300);
    }
  }, 100);
}

// 刪除標籤
function deleteLabel(index) {
  if (confirm(`確定要刪除「${labels[index]}」標籤和所有相關樣本嗎？`)) {
    const label = labels[index];
    delete samples[label];
    labels.splice(index, 1);
    renderLabels();
    showToast(`已刪除「${label}」標籤`);
    saveToLocalStorage();
  }
}

// 每幀呼叫，更新 lastLandmarks 且處理辨識或收集
function handleFrame(processedHandData) {
  if (model && recognizing && processedHandData) {
    const vector = landmarksToVector(processedHandData);
    if (vector.length !== 126) {
      console.error("Input vector size is not 126, it's: ", vector.length);
      return;
    }
    try {
      const input = tf.tensor([vector]);
      const probs = model.predict(input);
      const scores = probs.arraySync()[0];
      const maxIdx = scores.indexOf(Math.max(...scores));
      const pred = labels[maxIdx];
      const confidence = scores[maxIdx];
      
      updatePredictionDisplay(pred, confidence);
      recognizeGesture(pred);
      
      input.dispose();
      probs.dispose();
    } catch (error) {
      console.error('預測錯誤:', error);
    }
  }
}

// 更新預測顯示
function updatePredictionDisplay(prediction, confidence) {
  predictionOverlay.style.display = 'block';
  predictionOverlay.textContent = `${prediction} (${(confidence * 100).toFixed(1)}%)`;
  predictionOverlay.style.backgroundColor = `rgba(66, 133, 244, ${confidence * 0.8 + 0.2})`;
}

// 模型訓練
async function trainModel() {
  if (labels.length < 2) {
    return showToast('至少需要兩個標籤才能訓練模型', 'error');
  }
  
  let totalSamplesCount = 0; // Renamed from totalSamples to avoid conflict with global
  for (const label of labels) {
    if (!samples[label] || samples[label].length < 10) {
      return showToast(`「${label}」至少需要10個樣本（目前${samples[label] ? samples[label].length : 0}個）`, 'error');
    }
    totalSamplesCount += samples[label].length;
  }
  
  if (isTraining) {
    return showToast('模型訓練中，請稍候', 'warning');
  }
  
  let errorOccurred = false; // Flag to track if a data validation error occurred

  try {
    isTraining = true;
    showToast('開始訓練模型...');
    trainingStatus.classList.remove('hidden');
    progressBar.style.width = '0%'; // Reset progress bar
    epochInfo.textContent = '準備中...';
    
    const epochs = parseInt(document.getElementById('training-epochs').value) || 50;
    const learningRate = parseFloat(document.getElementById('learning-rate').value) || 0.001;
    const validationSplitRatio = parseFloat(document.getElementById('validation-split-ratio').value) || 0.2;
    
    const allSamples = [];
    const allLabelsNumeric = []; // Renamed from allLabels
    labels.forEach((lbl, i) => {
      if (samples[lbl]) {
        samples[lbl].forEach((vec, vecIndex) => {
          if (!Array.isArray(vec) || vec.length !== 126) {
            const errorMsg = `資料錯誤：標籤 "${lbl}" 的第 ${vecIndex + 1} 個樣本向量長度為 ${vec.length}，應為 126。`;
            console.error(errorMsg, "問題向量:", vec);
            showToast(errorMsg + "請檢查資料收集過程或清除問題標籤的樣本後重試。", 'error', 7000);
            errorOccurred = true;
            throw new Error(errorMsg); 
          }
          for (let j = 0; j < vec.length; j++) {
            if (typeof vec[j] !== 'number' || !isFinite(vec[j])) {
              const errorMsg = `資料錯誤：標籤 "${lbl}" 的第 ${vecIndex + 1} 個樣本中，索引 ${j} 處包含無效數值: ${vec[j]}。`;
              console.error(errorMsg, "問題向量:", vec);
              showToast(errorMsg + "請檢查資料收集過程或清除問題標籤的樣本後重試。", 'error', 7000);
              errorOccurred = true;
              throw new Error(errorMsg);
            }
          }
          allSamples.push(vec);
          allLabelsNumeric.push(i);
        });
      }
    });

    // Combine samples and labels for shuffling
    let combinedData = allSamples.map((sample, index) => ({sample, label: allLabelsNumeric[index]}));
    
    // Shuffle the combined data (tf.util.shuffle shuffles in place)
    tf.util.shuffle(combinedData);

    // Separate back into samples and labels
    const finalSamples = combinedData.map(item => item.sample);
    const finalLabelsNumeric = combinedData.map(item => item.label);

    if (errorOccurred) { // If a data validation error was thrown, stop here
        isTraining = false;
        trainingStatus.classList.add('hidden');
        return;
    }

    if (finalSamples.length === 0) {
        showToast('沒有有效的樣本可供訓練。', 'error');
        isTraining = false;
        trainingStatus.classList.add('hidden');
        return;
    }

    console.log(`準備訓練的總樣本數: ${finalSamples.length}`);
    if (finalSamples.length > 0) {
        console.log(`第一個隨機樣本向量 (前10個元素):`, finalSamples[0].slice(0, 10));
        console.log(`預期每個樣本的特徵數: 126`);
    }
    
    const xs = tf.tensor2d(finalSamples);
    const ys = tf.oneHot(tf.tensor1d(finalLabelsNumeric, 'int32'), labels.length);
    
    model = tf.sequential();
    model.add(tf.layers.dense({inputShape: [126], units: 256, activation: 'relu'}));
    model.add(tf.layers.dropout({rate: 0.3}));
    model.add(tf.layers.dense({units: 128, activation: 'relu'}));
    model.add(tf.layers.dropout({rate: 0.3}));
    model.add(tf.layers.dense({units: 128, activation: 'relu'}));
    model.add(tf.layers.dropout({rate: 0.3}));
    model.add(tf.layers.dense({units: labels.length, activation: 'softmax'}));
    
    model.compile({
      optimizer: tf.train.adam(learningRate),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });
    
    epochInfo.textContent = `Epoch: 0/${epochs}`;
    
    const history = await model.fit(xs, ys, {
      epochs: epochs,
      batchSize: 16,
      shuffle: true,
      validationSplit: validationSplitRatio, // Use the value from the input
      callbacks: {
        onEpochBegin: (epoch) => {
          epochInfo.textContent = `Epoch: ${epoch+1}/${epochs}`;
        },
        onEpochEnd: (epoch, logs) => {
          const valAccuracy = logs.val_acc; // Corrected key to val_acc
          const trainAccuracy = logs.acc || logs.accuracy;
          let displayAccuracy;
          let accuracyLabel = "";

          if (typeof valAccuracy === 'number') {
            displayAccuracy = valAccuracy;
            accuracyLabel = "驗證準確率";
          } else if (typeof trainAccuracy === 'number') {
            displayAccuracy = trainAccuracy;
            accuracyLabel = "訓練準確率";
          }

          progressBar.style.width = `${((epoch+1) / epochs) * 100}%`;
          if (typeof displayAccuracy === 'number') {
            epochInfo.textContent = `Epoch: ${epoch+1}/${epochs} (${accuracyLabel}: ${(displayAccuracy * 100).toFixed(1)}%)`;
            console.log(`Epoch ${epoch+1}: ${accuracyLabel} = ${(displayAccuracy * 100).toFixed(1)}%`);
          } else {
            epochInfo.textContent = `Epoch: ${epoch+1}/${epochs}`;
            console.log(`Epoch ${epoch+1}: 準確率資訊不可用`);
          }
        }
      }
    });
    
    xs.dispose();
    ys.dispose();

    const accuracyLabelElement = document.getElementById('val_accuracy');
    if (accuracyLabelElement) {
        if (history.history.val_acc && history.history.val_acc.length > 0) { // Corrected key to val_acc
            const finalValAccuracy = history.history.val_acc[epochs - 1]; // Corrected key to val_acc
            if (typeof finalValAccuracy === 'number') {
                accuracyLabelElement.textContent = `驗證集準確率：${(finalValAccuracy * 100).toFixed(2)}%`;
            } else {
                accuracyLabelElement.textContent = '驗證集準確率：數據錯誤';
            }
        } else {
            accuracyLabelElement.textContent = '驗證集準確率：N/A (未啟用或無數據)';
            console.log('Debug: history.history content:', history.history); // Log history if val_accuracy is missing
        }
    }
    
    showToast('模型訓練完成！', 'success');
    saveToLocalStorage(); // Save labels and samples, model is saved separately
  } catch (error) {
    console.error('訓練失敗:', error);
    if (!errorOccurred) { // Only show generic error if not a data validation error
        showToast(error.message && error.message.startsWith('資料錯誤：') ? error.message : '模型訓練失敗，詳情請查看控制台。', 'error', 7000);
    }
  } finally {
    isTraining = false;
    // Hide training status only if no data validation error occurred or if it was hidden by it
    if (!errorOccurred && trainingStatus && !trainingStatus.classList.contains('hidden')) {
        trainingStatus.classList.add('hidden');
    } else if (errorOccurred && trainingStatus) { // Ensure it's hidden if data error
        trainingStatus.classList.add('hidden');
    }
  }
}

// 儲存模型
async function saveModel() {
  if (!model) {
    return showToast('尚未有模型可儲存', 'error');
  }
  
  try {
    showToast('正在儲存模型...');
    await model.save('downloads://hand_gesture_model');
    showToast('模型儲存完成', 'success');
  } catch (error) {
    console.error('儲存模型失敗:', error);
    showToast('儲存模型失敗', 'error');
  }
}

// 載入模型
async function loadModel(e) {
  const file = e.target.files[0];
  if (!file) return;
  
  try {
    showToast('正在載入模型...');
    model = await tf.loadLayersModel(tf.io.browserFiles([file]));
    showToast('模型載入完成', 'success');
  } catch (error) {
    console.error('載入模型失敗:', error);
    showToast('載入模型失敗', 'error');
  }
}

// 切換辨識狀態
function toggleRecognition() {
  const btn = document.getElementById('start-recognize');
  
  if (recognizing) {
    recognizing = false;
    btn.innerHTML = '<span class="material-icons">play_circle</span> 啟動辨識';
    predictionOverlay.style.display = 'none';
    showToast('辨識已停止');
  } else {
    if (!model) {
      return showToast('請先訓練或載入模型', 'error');
    }
    
    recognizing = true;
    btn.innerHTML = '<span class="material-icons">stop_circle</span> 停止辨識';
    gestureStart = null;
    showToast('辨識已啟動');
  }
}

// 重置手勢狀態
function resetGesture() {
  gestureStart = null;
  currentGesture = null;
}

// 手勢持續判定與文字輸出
function recognizeGesture(pred) {
  const threshold = parseFloat(document.getElementById('hold-threshold').value) * 1000;
  const now = Date.now();
  
  if (pred !== currentGesture) {
    currentGesture = pred;
    gestureStart = now;
  } else if (gestureStart && now - gestureStart >= threshold) {
    appendText(pred);
    gestureStart = null;
    showToast(`已辨識: ${pred}`, 'success', 800);
  }
}

// 文字輸出
function appendText(text) {
  const ta = document.getElementById('text-output');
  ta.value = ta.value + text + ' ';
  ta.focus();
  ta.selectionStart = ta.selectionEnd = ta.value.length;
}

// 儲存到本地儲存
function saveToLocalStorage() {
  try {
    const data = {
      labels: labels,
      samples: samples
    };
    localStorage.setItem('handGestureData', JSON.stringify(data));
  } catch (error) {
    console.error('儲存本地資料失敗:', error);
  }
}

// 從本地儲存載入
function loadFromLocalStorage() {
  try {
    const dataStr = localStorage.getItem('handGestureData');
    if (dataStr) {
      const data = JSON.parse(dataStr);
      labels = data.labels || [];
      samples = data.samples || {};
      renderLabels();
      showToast('從本地儲存載入資料');
    }
  } catch (error) {
    console.error('載入本地資料失敗:', error);
  }
}

// 顯示提示訊息
function showToast(message, type = 'info', duration = 2000) {
  const toastEl = document.getElementById('toast');
  toastEl.textContent = message;
  
  if (type === 'error') {
    toastEl.style.backgroundColor = '#ea4335';
  } else if (type === 'success') {
    toastEl.style.backgroundColor = '#34a853';
  } else if (type === 'warning') {
    toastEl.style.backgroundColor = '#fbbc04';
    toastEl.style.color = '#000';
  } else {
    toastEl.style.backgroundColor = '#4285f4';
  }
  
  toastEl.style.opacity = '1';
  
  setTimeout(() => {
    toastEl.style.opacity = '0';
  }, duration);
}
