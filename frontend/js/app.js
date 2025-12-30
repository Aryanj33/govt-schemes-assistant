/**
 * Scholarship Voice Assistant - Main Application
 * Handles audio recording, API communication, and playback
 */

class ScholarshipVoiceApp {
    constructor() {
        // Configuration
        this.config = {
            apiUrl: window.location.hostname === 'localhost'
                ? 'http://localhost:8080'
                : window.location.origin,
            sampleRate: 16000,
            channels: 1,
        };

        // State
        this.mediaRecorder = null;
        this.audioContext = null;
        this.analyser = null;
        this.audioChunks = [];
        this.isRecording = false;
        this.isConnected = false;

        // UI Controller
        this.ui = null;

        // Animation frame for visualizer
        this.animationFrame = null;

        // Bind methods
        this.startRecording = this.startRecording.bind(this);
        this.stopRecording = this.stopRecording.bind(this);

        // Initialize when DOM is ready
        this.init();
    }

    async init() {
        console.log('🚀 Initializing Scholarship Voice Assistant...');

        // Wait for DOM
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.setup());
        } else {
            this.setup();
        }
    }

    async setup() {
        // Initialize UI controller
        this.ui = new window.UIController();

        // Check connection to backend
        await this.checkConnection();

        // Request microphone permission early
        await this.requestMicrophonePermission();

        // Setup text input handlers
        this.setupTextInput();

        console.log('✅ App initialized');
    }

    /**
     * Setup text input event handlers
     */
    setupTextInput() {
        const textInput = document.getElementById('text-input');
        const sendButton = document.getElementById('send-button');

        if (textInput && sendButton) {
            // Send on Enter key
            textInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendTextMessage();
                }
            });

            // Send on button click
            sendButton.addEventListener('click', () => {
                this.sendTextMessage();
            });

            console.log('📝 Text input handlers ready');
        }
    }

    /**
     * Send text message to backend
     */
    async sendTextMessage() {
        const textInput = document.getElementById('text-input');
        const sendButton = document.getElementById('send-button');
        const message = textInput.value.trim();

        if (!message) return;
        if (!this.isConnected) {
            this.ui.showToast('Not connected to server. Please start the backend.');
            return;
        }

        // Clear input and disable
        textInput.value = '';
        sendButton.disabled = true;

        // Add user message to UI
        this.ui.addMessage('user', message);

        // Show typing indicator
        this.showTypingIndicator();

        try {
            const response = await fetch(`${this.config.apiUrl}/text`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: message })
            });

            // Remove typing indicator
            this.hideTypingIndicator();

            if (response.ok) {
                const data = await response.json();
                this.ui.addMessage('assistant', data.response);

                // Log session state for debugging
                if (data.session_state) {
                    console.log('📊 Session state:', data.session_state);
                }
            } else {
                throw new Error(`Server error: ${response.status}`);
            }
        } catch (error) {
            console.error('❌ Text send failed:', error);
            this.hideTypingIndicator();
            this.ui.addMessage('assistant', 'Sorry, कुछ problem हो गई। Please try again.');
            this.ui.showToast('Failed to get response');
        }

        sendButton.disabled = false;
        textInput.focus();
    }

    /**
     * Show typing indicator
     */
    showTypingIndicator() {
        const container = document.getElementById('transcript-container');
        const indicator = document.createElement('div');
        indicator.id = 'typing-indicator';
        indicator.className = 'message assistant';
        indicator.innerHTML = `
            <div class="message-label">🤖 Vidya</div>
            <div class="typing-indicator">
                <span></span><span></span><span></span>
            </div>
        `;
        container.appendChild(indicator);
        container.scrollTop = container.scrollHeight;
    }

    /**
     * Hide typing indicator
     */
    hideTypingIndicator() {
        const indicator = document.getElementById('typing-indicator');
        if (indicator) {
            indicator.remove();
        }
    }

    /**
     * Check connection to backend server
     */
    async checkConnection() {
        this.ui.setConnectionStatus('connecting', 'Connecting...');

        try {
            const response = await fetch(`${this.config.apiUrl}/health`, {
                method: 'GET',
                headers: { 'Accept': 'application/json' }
            });

            if (response.ok) {
                const data = await response.json();
                this.isConnected = true;
                this.ui.setConnectionStatus('connected', `Connected (${data.scholarships_loaded} scholarships)`);
                console.log('✅ Connected to backend:', data);
            } else {
                throw new Error('Server not responding');
            }
        } catch (error) {
            console.error('❌ Connection failed:', error);
            this.isConnected = false;
            this.ui.setConnectionStatus('error', 'Offline - Start backend server');
            this.ui.showToast('Backend server not running. Start with: python backend/main.py');
        }
    }

    /**
     * Request microphone permission
     */
    async requestMicrophonePermission() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            // Stop immediately, we just wanted permission
            stream.getTracks().forEach(track => track.stop());
            console.log('🎤 Microphone permission granted');
            return true;
        } catch (error) {
            console.error('❌ Microphone permission denied:', error);
            this.ui.showToast('Microphone access denied. Please allow microphone access.');
            this.ui.disableMic();
            return false;
        }
    }

    /**
     * Start recording audio
     */
    async startRecording() {
        if (this.isRecording) return;

        try {
            // Get audio stream
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: this.config.sampleRate,
                    channelCount: this.config.channels,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });

            // Setup audio context for visualization
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const source = this.audioContext.createMediaStreamSource(stream);
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 256;
            source.connect(this.analyser);

            // Start visualizer animation
            this.startVisualizerAnimation();

            // Setup media recorder
            const mimeType = this.getSupportedMimeType();
            this.mediaRecorder = new MediaRecorder(stream, { mimeType });
            this.audioChunks = [];

            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };

            this.mediaRecorder.onstop = () => {
                this.processRecording();
            };

            // Start recording
            this.mediaRecorder.start(100); // Collect data every 100ms
            this.isRecording = true;

            console.log('🎙️ Recording started');

        } catch (error) {
            console.error('❌ Failed to start recording:', error);
            this.ui.showToast('Failed to access microphone');
            this.ui.stopListening();
        }
    }

    /**
     * Stop recording audio
     */
    stopRecording() {
        if (!this.isRecording || !this.mediaRecorder) return;

        this.isRecording = false;
        this.mediaRecorder.stop();

        // Stop visualizer
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
        }

        // Stop audio tracks
        if (this.mediaRecorder.stream) {
            this.mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }

        // Close audio context
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }

        console.log('⏹️ Recording stopped');
    }

    /**
     * Process recorded audio and send to backend
     */
    async processRecording() {
        if (this.audioChunks.length === 0) {
            console.log('⚠️ No audio recorded');
            return;
        }

        this.ui.showLoading();

        try {
            // Create blob from chunks
            const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });

            // Convert to WAV for better compatibility with Whisper
            const wavBlob = await this.convertToWav(audioBlob);

            console.log(`📤 Sending ${wavBlob.size} bytes of audio`);

            // Send to backend
            const response = await fetch(`${this.config.apiUrl}/audio`, {
                method: 'POST',
                body: wavBlob,
                headers: {
                    'Content-Type': 'audio/wav'
                }
            });

            if (response.ok) {
                // Get response audio
                const responseBlob = await response.blob();

                // Add placeholder messages (in real impl, get transcript from response headers)
                this.ui.addMessage('user', '[Your speech was processed]');

                // Play response audio
                await this.playAudio(responseBlob);

                this.ui.addMessage('assistant', '[Response played - see console for details]');
            } else {
                throw new Error(`Server error: ${response.status}`);
            }

        } catch (error) {
            console.error('❌ Processing failed:', error);
            this.ui.showToast('Failed to process audio. Please try again.');

            // Fallback to text mode
            this.fallbackToTextMode();
        }

        this.ui.hideLoading();
    }

    /**
     * Fallback to text input when audio fails
     */
    async fallbackToTextMode() {
        const userInput = prompt('Voice failed. Type your question (Hindi/English):');

        if (!userInput) return;

        this.ui.addMessage('user', userInput);
        this.ui.showLoading();

        try {
            const response = await fetch(`${this.config.apiUrl}/text`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: userInput })
            });

            if (response.ok) {
                const data = await response.json();
                this.ui.addMessage('assistant', data.response);
            }
        } catch (error) {
            console.error('Text mode error:', error);
        }

        this.ui.hideLoading();
    }

    /**
     * Convert audio blob to WAV format
     */
    async convertToWav(blob) {
        // For simplicity, we'll send webm directly
        // In production, use audiobuffer-to-wav library
        // The backend should handle webm via ffmpeg

        // Try to convert using AudioContext
        try {
            const arrayBuffer = await blob.arrayBuffer();
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

            // Convert to WAV
            const wavData = this.audioBufferToWav(audioBuffer);
            return new Blob([wavData], { type: 'audio/wav' });

        } catch (error) {
            console.log('WAV conversion failed, sending original format');
            return blob;
        }
    }

    /**
     * Convert AudioBuffer to WAV format
     */
    audioBufferToWav(buffer) {
        const numChannels = buffer.numberOfChannels;
        const sampleRate = buffer.sampleRate;
        const format = 1; // PCM
        const bitDepth = 16;

        const bytesPerSample = bitDepth / 8;
        const blockAlign = numChannels * bytesPerSample;

        const samples = buffer.length;
        const dataSize = samples * blockAlign;
        const bufferSize = 44 + dataSize;

        const arrayBuffer = new ArrayBuffer(bufferSize);
        const view = new DataView(arrayBuffer);

        // WAV header
        this.writeString(view, 0, 'RIFF');
        view.setUint32(4, 36 + dataSize, true);
        this.writeString(view, 8, 'WAVE');
        this.writeString(view, 12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, format, true);
        view.setUint16(22, numChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * blockAlign, true);
        view.setUint16(32, blockAlign, true);
        view.setUint16(34, bitDepth, true);
        this.writeString(view, 36, 'data');
        view.setUint32(40, dataSize, true);

        // Audio data
        const offset = 44;
        const channelData = buffer.getChannelData(0);

        for (let i = 0; i < samples; i++) {
            const sample = Math.max(-1, Math.min(1, channelData[i]));
            const intSample = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
            view.setInt16(offset + i * 2, intSample, true);
        }

        return arrayBuffer;
    }

    writeString(view, offset, string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }

    /**
     * Play audio response with Barge-In support
     */
    async playAudio(blob) {
        return new Promise((resolve, reject) => {
            // Stop any existing audio
            if (this.currentAudio) {
                this.currentAudio.pause();
                this.currentAudio = null;
            }

            const audio = new Audio();
            audio.src = URL.createObjectURL(blob);
            this.currentAudio = audio;

            // Unlock audio on iOS/mobile
            audio.play().catch(e => console.log('Autoplay prevented:', e));

            // Start monitoring for interruption (Barge-In)
            this.startInterruptionMonitoring(audio, resolve);

            audio.onended = () => {
                this.stopInterruptionMonitoring();
                URL.revokeObjectURL(audio.src);
                this.currentAudio = null;
                resolve();
            };

            audio.onerror = (error) => {
                this.stopInterruptionMonitoring();
                URL.revokeObjectURL(audio.src);
                this.currentAudio = null;
                reject(error);
            };
        });
    }

    /**
     * Start monitoring microphone for interruption
     */
    async startInterruptionMonitoring(audioElement, resolvePromise) {
        if (this.isMonitoringInterruption) return;

        try {
            console.log('👂 Barge-in monitoring started...');
            this.isMonitoringInterruption = true;
            this.ui.showToast('Speak to interrupt...');

            // We need a separate stream for monitoring to avoid conflict with recording
            // or we can reuse if we manage state carefully. 
            // For simplicity, we get a new stream or reuse existing if available.

            let stream = null;
            if (this.mediaRecorder && this.mediaRecorder.stream && this.mediaRecorder.stream.active) {
                stream = this.mediaRecorder.stream;
            } else {
                stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            }

            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const source = audioContext.createMediaStreamSource(stream);
            const analyser = audioContext.createAnalyser();
            analyser.fftSize = 256;
            source.connect(analyser);

            const dataArray = new Uint8Array(analyser.frequencyBinCount);

            // Interruption threshold (tunable)
            const THRESHOLD = 30; // 0-255
            const SUSTAINED_FRAMES = 3; // How many frames to confirm voice
            let triggers = 0;

            const checkVolume = () => {
                if (!this.isMonitoringInterruption) {
                    audioContext.close();
                    return;
                }

                analyser.getByteFrequencyData(dataArray);

                // Calculate average volume
                let sum = 0;
                for (let i = 0; i < dataArray.length; i++) {
                    sum += dataArray[i];
                }
                const average = sum / dataArray.length;

                // Detect voice
                if (average > THRESHOLD) {
                    triggers++;
                } else {
                    triggers = Math.max(0, triggers - 1);
                }

                if (triggers >= SUSTAINED_FRAMES) {
                    console.log('🛑 Interruption detected! Stopping playback.');

                    // Stop audio
                    audioElement.pause();
                    audioElement.currentTime = 0;

                    // Cleanup monitoring
                    this.isMonitoringInterruption = false;
                    audioContext.close();

                    // Trigger new recording immediately
                    this.ui.showToast('Interrupted! Listening...');
                    this.startRecording(); // Start full recording

                    // Resolve the playback promise early
                    resolvePromise();
                    return;
                }

                requestAnimationFrame(checkVolume);
            };

            checkVolume();

        } catch (error) {
            console.error('Barge-in setup failed:', error);
        }
    }

    stopInterruptionMonitoring() {
        this.isMonitoringInterruption = false;
        console.log('👂 Barge-in monitoring stopped');
    }

    /**
     * Start visualizer animation
     */
    startVisualizerAnimation() {
        if (!this.analyser) return;

        const dataArray = new Uint8Array(this.analyser.frequencyBinCount);

        const animate = () => {
            if (!this.isRecording) return;

            this.analyser.getByteFrequencyData(dataArray);
            this.ui.updateVisualizer(dataArray);

            this.animationFrame = requestAnimationFrame(animate);
        };

        animate();
    }

    /**
     * Get supported audio MIME type
     */
    getSupportedMimeType() {
        const types = [
            'audio/webm;codecs=opus',
            'audio/webm',
            'audio/ogg;codecs=opus',
            'audio/mp4'
        ];

        for (const type of types) {
            if (MediaRecorder.isTypeSupported(type)) {
                console.log(`📼 Using MIME type: ${type}`);
                return type;
            }
        }

        console.log('📼 Using default MIME type');
        return '';
    }
}

// Initialize app globally
window.app = new ScholarshipVoiceApp();
