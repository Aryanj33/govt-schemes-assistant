/**
 * Scholarship Voice Assistant - UI Controller
 * Manages UI state, animations, and interactions
 */

class UIController {
    constructor() {
        // DOM Elements
        this.micButton = document.getElementById('mic-button');
        this.micLabel = document.getElementById('mic-label');
        this.statusIndicator = document.getElementById('status-indicator');
        this.statusText = document.getElementById('status-text');
        this.visualizer = document.getElementById('visualizer');
        this.transcriptContainer = document.getElementById('transcript-container');
        this.toast = document.getElementById('toast');

        // State
        this.isListening = false;
        this.toastTimeout = null;

        // Bind methods
        this.onMicClick = this.onMicClick.bind(this);
        this.handleKeyboard = this.handleKeyboard.bind(this);

        // Initialize
        this.init();
    }

    init() {
        // Add event listeners
        this.micButton.addEventListener('click', this.onMicClick);
        document.addEventListener('keydown', this.handleKeyboard);

        console.log('🎨 UI Controller initialized');
    }

    /**
     * Handle microphone button click
     */
    onMicClick() {
        if (this.isListening) {
            this.stopListening();
        } else {
            this.startListening();
        }
    }

    /**
     * Handle keyboard shortcuts
     */
    handleKeyboard(event) {
        // Space bar to toggle listening
        if (event.code === 'Space' && event.target.tagName !== 'INPUT') {
            event.preventDefault();
            this.onMicClick();
        }

        // Escape to stop
        if (event.code === 'Escape' && this.isListening) {
            this.stopListening();
        }
    }

    /**
     * Start listening mode
     */
    startListening() {
        this.isListening = true;
        this.micButton.classList.add('active');
        this.visualizer.classList.add('active');
        this.micLabel.textContent = 'Listening... (Press Space or Click to stop)';

        // Trigger app.js to start recording
        if (window.app && window.app.startRecording) {
            window.app.startRecording();
        }
    }

    /**
     * Stop listening mode
     */
    stopListening() {
        this.isListening = false;
        this.micButton.classList.remove('active');
        this.visualizer.classList.remove('active');
        this.micLabel.textContent = 'Click to start speaking';

        // Trigger app.js to stop recording
        if (window.app && window.app.stopRecording) {
            window.app.stopRecording();
        }
    }

    /**
     * Update connection status
     * @param {string} status - 'connected', 'connecting', 'disconnected', 'error'
     * @param {string} text - Status text to display
     */
    setConnectionStatus(status, text) {
        // Remove all status classes
        this.statusIndicator.classList.remove('connected', 'connecting', 'error');

        // Add appropriate class
        if (status === 'connected') {
            this.statusIndicator.classList.add('connected');
        } else if (status === 'connecting') {
            this.statusIndicator.classList.add('connecting');
        } else if (status === 'error') {
            this.statusIndicator.classList.add('error');
        }

        this.statusText.textContent = text;
    }

    /**
     * Add a message to the transcript
     * @param {string} role - 'user' or 'assistant'
     * @param {string} content - Message content
     */
    addMessage(role, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;

        const labelDiv = document.createElement('div');
        labelDiv.className = 'message-label';
        labelDiv.textContent = role === 'user' ? '🎤 You' : '🤖 Vidya';

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = content;

        messageDiv.appendChild(labelDiv);
        messageDiv.appendChild(contentDiv);

        this.transcriptContainer.appendChild(messageDiv);

        // Scroll to bottom
        this.transcriptContainer.scrollTop = this.transcriptContainer.scrollHeight;
    }

    /**
     * Update the audio visualizer bars
     * @param {Uint8Array} dataArray - Audio frequency data
     */
    updateVisualizer(dataArray) {
        const bars = this.visualizer.querySelectorAll('.visualizer-bar');

        if (!dataArray || !bars.length) return;

        const step = Math.floor(dataArray.length / bars.length);

        bars.forEach((bar, index) => {
            const value = dataArray[index * step] || 0;
            const height = Math.max(10, (value / 255) * 40);
            bar.style.height = `${height}px`;
        });
    }

    /**
     * Show an error toast
     * @param {string} message - Error message
     */
    showToast(message) {
        this.toast.textContent = message;
        this.toast.classList.add('visible');

        // Clear existing timeout
        if (this.toastTimeout) {
            clearTimeout(this.toastTimeout);
        }

        // Hide after 4 seconds
        this.toastTimeout = setTimeout(() => {
            this.toast.classList.remove('visible');
        }, 4000);
    }

    /**
     * Show loading state
     */
    showLoading() {
        this.micLabel.innerHTML = '<span class="spinner"></span> Processing...';
    }

    /**
     * Hide loading state
     */
    hideLoading() {
        this.micLabel.textContent = 'Click to start speaking';
    }

    /**
     * Disable microphone button
     */
    disableMic() {
        this.micButton.disabled = true;
        this.micButton.style.opacity = '0.5';
        this.micButton.style.cursor = 'not-allowed';
    }

    /**
     * Enable microphone button
     */
    enableMic() {
        this.micButton.disabled = false;
        this.micButton.style.opacity = '1';
        this.micButton.style.cursor = 'pointer';
    }

    /**
     * Clear transcript history
     */
    clearTranscript() {
        this.transcriptContainer.innerHTML = '';
    }
}

// Export for use in app.js
window.UIController = UIController;
