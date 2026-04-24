document.addEventListener('DOMContentLoaded', function() {
    const scriptForm = document.getElementById('scriptForm');
    const generatingOverlay = document.getElementById('generatingOverlay');
    const generateBtn = document.getElementById('generateBtn');
    const genStep3 = document.getElementById('genStep3');
    
    const resultVideo = document.getElementById('resultVideo');
    if (resultVideo) {
        resultVideo.addEventListener('loadedmetadata', function() {
            const duration = formatDuration(resultVideo.duration);
            const durationEl = document.getElementById('videoDuration');
            if (durationEl) durationEl.textContent = duration;
        });
    }
    
    if (scriptForm) {
        scriptForm.addEventListener('submit', function(e) {
            const platformSelect = scriptForm.querySelector('select[name="type"]');
            const platform = platformSelect.value;
            const needsVideo = ['Reel', 'YouTube Short', 'TikTok'].includes(platform);
            const needsAudio = !['Post'].includes(platform);
            
            const genStep1 = document.getElementById('genStep1');
            const genStep2 = document.getElementById('genStep2');
            const genStep3 = document.getElementById('genStep3');
            
            genStep3.querySelector('span:last-child').textContent = needsVideo ? 'Video' : 'Finalize';
            
            if (generatingOverlay && generateBtn) {
                generatingOverlay.style.display = 'flex';
                generateBtn.disabled = true;
                
                genStep1.classList.add('active');
                genStep2.classList.remove('active');
                genStep3.classList.remove('active');
                
                setTimeout(() => {
                    genStep1.classList.remove('active');
                    if (needsAudio) genStep2.classList.add('active');
                }, needsAudio ? 1500 : 500);
                
                setTimeout(() => {
                    genStep2.classList.remove('active');
                    genStep3.classList.add('active');
                }, needsAudio ? (needsVideo ? 3000 : 2500) : 1500);
            }
        });
    }
});

function formatDuration(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return mins > 0 ? `${mins}:${secs.toString().padStart(2, '0')}` : `${secs}s`;
}

function copyToClipboard() {
    const outputContent = document.querySelector('.result-content');
    if (outputContent) {
        navigator.clipboard.writeText(outputContent.textContent).then(function() {
            const copyBtn = document.querySelector('.copy-btn');
            if (copyBtn) {
                const originalHTML = copyBtn.innerHTML;
                copyBtn.innerHTML = '<svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg> Copied!';
                copyBtn.style.background = 'rgba(34, 197, 94, 0.15)';
                copyBtn.style.borderColor = 'rgba(34, 197, 94, 0.3)';
                copyBtn.style.color = '#22c55e';
                setTimeout(function() {
                    copyBtn.innerHTML = originalHTML;
                    copyBtn.style.background = '';
                    copyBtn.style.borderColor = '';
                    copyBtn.style.color = '';
                }, 2000);
            }
        });
    }
}

function togglePassword(button) {
    const wrapper = button.closest('.password-wrapper');
    const input = wrapper.querySelector('input[type="password"], input[type="text"]');
    const isPassword = input.type === 'password';
    
    input.type = isPassword ? 'text' : 'password';
    button.classList.toggle('active', isPassword);
}