document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const videoDropZone = document.getElementById('videoDropZone');
    const videoInput = document.getElementById('videoInput');
    const videoPreview = document.getElementById('videoPreview');
    const videoPlayer = document.getElementById('videoPlayer');
    const videoName = document.getElementById('videoName');
    const removeVideo = document.getElementById('removeVideo');
    
    const disclaimerDropZone = document.getElementById('disclaimerDropZone');
    const disclaimerInput = document.getElementById('disclaimerInput');
    const disclaimerPreview = document.getElementById('disclaimerPreview');
    const disclaimerImage = document.getElementById('disclaimerImage');
    const disclaimerName = document.getElementById('disclaimerName');
    const removeDisclaimer = document.getElementById('removeDisclaimer');
    
    const framesPerSecond = document.getElementById('framesPerSecond');
    const framesValue = document.getElementById('framesValue');
    const modelEndpoint = document.getElementById('modelEndpoint');
    
    const step1 = document.getElementById('step1');
    const step2 = document.getElementById('step2');
    const step3 = document.getElementById('step3');
    const step4 = document.getElementById('step4');
    
    const processButton = document.getElementById('processButton');
    const processingStatus = document.getElementById('processingStatus');
    const progressBar = document.getElementById('progressBar');
    const progressPercent = document.getElementById('progressPercent');
    const currentStep = document.getElementById('currentStep');
    const statusText = document.getElementById('statusText');
    
    const errorAlert = document.getElementById('errorAlert');
    const errorMessage = document.getElementById('errorMessage');
    const closeError = document.getElementById('closeError');
    
    const totalFrames = document.getElementById('totalFrames');
    const smokingFrames = document.getElementById('smokingFrames');
    const framesPerSecondResult = document.getElementById('framesPerSecondResult');
    const fileSize = document.getElementById('fileSize');
    const downloadButton = document.getElementById('downloadButton');
    const resultVideo = document.getElementById('resultVideo');
    
    // State variables
    let videoFile = null;
    let disclaimerFile = null;
    let isProcessing = false;
    let processingComplete = false;
    let currentTaskId = null;
    let statusCheckInterval = null;
    
    // API endpoints
    const API_PROCESS = '/api/process';
    const API_STATUS = '/api/status/';
    const API_DOWNLOAD = '/api/download/';
    
    // Initialize UI state
    updateUIState();
    
    // Event Listeners
    
    // Video Drop Zone
    videoDropZone.addEventListener('click', () => videoInput.click());
    videoDropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        videoDropZone.classList.add('active');
    });
    videoDropZone.addEventListener('dragleave', () => {
        videoDropZone.classList.remove('active');
    });
    videoDropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        videoDropZone.classList.remove('active');
        
        if (e.dataTransfer.files.length) {
            handleVideoFile(e.dataTransfer.files[0]);
        }
    });
    
    videoInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleVideoFile(e.target.files[0]);
        }
    });
    
    removeVideo.addEventListener('click', () => {
        videoFile = null;
        videoPlayer.src = '';
        videoPreview.classList.add('hidden');
        updateUIState();
    });
    
    // Disclaimer Drop Zone
    disclaimerDropZone.addEventListener('click', () => disclaimerInput.click());
    disclaimerDropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        disclaimerDropZone.classList.add('active');
    });
    disclaimerDropZone.addEventListener('dragleave', () => {
        disclaimerDropZone.classList.remove('active');
    });
    disclaimerDropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        disclaimerDropZone.classList.remove('active');
        
        if (e.dataTransfer.files.length) {
            handleDisclaimerFile(e.dataTransfer.files[0]);
        }
    });
    
    disclaimerInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleDisclaimerFile(e.target.files[0]);
        }
    });
    
    removeDisclaimer.addEventListener('click', () => {
        disclaimerFile = null;
        disclaimerImage.src = '';
        disclaimerPreview.classList.add('hidden');
        updateUIState();
    });
    
    // Frames per second slider
    framesPerSecond.addEventListener('input', () => {
        framesValue.textContent = framesPerSecond.value;
    });
    
    // Process button
    processButton.addEventListener('click', () => {
        if (videoFile && !isProcessing) {
            startProcessing();
        }
    });
    
    // Close error button
    closeError.addEventListener('click', () => {
        errorAlert.classList.add('hidden');
    });
    
    // Download button
    downloadButton.addEventListener('click', () => {
        if (currentTaskId) {
            window.location.href = API_DOWNLOAD + currentTaskId;
        }
    });
    
    // Functions
    
    function handleVideoFile(file) {
        // Check if file is a video
        if (!file.type.startsWith('video/')) {
            showError('Please upload a valid video file.');
            return;
        }
        
        videoFile = file;
        videoName.textContent = file.name;
        
        // Create object URL for preview
        const objectUrl = URL.createObjectURL(file);
        videoPlayer.src = objectUrl;
        videoPreview.classList.remove('hidden');
        
        updateUIState();
    }
    
    function handleDisclaimerFile(file) {
        // Check if file is an image
        if (!file.type.startsWith('image/')) {
            showError('Please upload a valid image file.');
            return;
        }
        
        disclaimerFile = file;
        disclaimerName.textContent = file.name;
        
        // Create object URL for preview
        const objectUrl = URL.createObjectURL(file);
        disclaimerImage.src = objectUrl;
        disclaimerPreview.classList.remove('hidden');
        
        updateUIState();
    }
    
    function updateUIState() {
        // Update step 2 (settings)
        if (videoFile) {
            step2.classList.remove('opacity-50');
        } else {
            step2.classList.add('opacity-50');
        }
        
        // Update step 3 (process)
        if (videoFile && !isProcessing) {
            step3.classList.remove('opacity-50');
            processButton.disabled = false;
        } else {
            step3.classList.add('opacity-50');
            processButton.disabled = true;
        }
        
        // Update frames per second result
        framesPerSecondResult.textContent = `• Frames per second analyzed: ${framesPerSecond.value}`;
    }
    
    function showError(message) {
        errorMessage.textContent = message;
        errorAlert.classList.remove('hidden');
        
        // Auto-hide after 5 seconds
        setTimeout(() => {
            errorAlert.classList.add('hidden');
        }, 5000);
    }
    
    function startProcessing() {
        isProcessing = true;
        processingStatus.classList.remove('hidden');
        updateUIState();
        
        // Create FormData
        const formData = new FormData();
        formData.append('video', videoFile);
        formData.append('frames_per_second', framesPerSecond.value);
        formData.append('model_endpoint', modelEndpoint.value);
        
        if (disclaimerFile) {
            formData.append('disclaimer_image', disclaimerFile);
        }
        
        // Send request to API
        fetch(API_PROCESS, {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to start processing');
            }
            return response.json();
        })
        .then(data => {
            if (data.task_id) {
                currentTaskId = data.task_id;
                updateProgress('Initializing...', 0);
                
                // Start checking status
                statusCheckInterval = setInterval(checkStatus, 1000);
            } else {
                throw new Error('Invalid response from server');
            }
        })
        .catch(error => {
            isProcessing = false;
            updateUIState();
            showError('Error: ' + error.message);
        });
    }
    
    function checkStatus() {
        if (!currentTaskId) {
            clearInterval(statusCheckInterval);
            return;
        }
        
        fetch(API_STATUS + currentTaskId)
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to get status');
            }
            return response.json();
        })
        .then(data => {
            // Update progress
            updateProgress(getStepText(data.status), data.progress, data.message);
            
            // Check if processing is complete
            if (data.status === 'completed') {
                completeProcessing(data);
            } else if (data.status === 'error') {
                showError('Processing error: ' + data.message);
                isProcessing = false;
                updateUIState();
                clearInterval(statusCheckInterval);
            }
        })
        .catch(error => {
            console.error('Error checking status:', error);
        });
    }
    
    function getStepText(status) {
        switch (status) {
            case 'initializing':
                return 'Initializing...';
            case 'extracting_frames':
                return 'Extracting frames...';
            case 'detecting_smoking':
                return 'Detecting smoking...';
            case 'adding_disclaimers':
                return 'Adding disclaimers to video...';
            case 'completed':
                return 'Processing complete!';
            case 'error':
                return 'Error';
            default:
                return 'Processing...';
        }
    }
    
    function updateProgress(step, percent, statusMessage = '') {
        currentStep.textContent = step;
        progressBar.style.width = `${percent}%`;
        progressPercent.textContent = `${Math.round(percent)}%`;
        
        if (statusMessage) {
            statusText.textContent = statusMessage;
        }
    }
    
    function completeProcessing(data) {
        // Clear interval
        clearInterval(statusCheckInterval);
        
        isProcessing = false;
        processingComplete = true;
        
        // Hide processing status
        processingStatus.classList.add('hidden');
        
        // Show results
        step4.classList.remove('hidden');
        
        // Update results
        totalFrames.textContent = `• Total frames analyzed: ${data.total_frames}`;
        smokingFrames.textContent = `• Smoking detected in: ${data.smoking_frames} frames (${data.smoking_percentage}%)`;
        framesPerSecondResult.textContent = `• Frames per second analyzed: ${data.frames_per_second}`;
        fileSize.textContent = `• Output file size: ${data.file_size}`;
        
        // Set video source
        resultVideo.src = API_DOWNLOAD + currentTaskId;
        
        // Update UI state
        updateUIState();
    }
}); 