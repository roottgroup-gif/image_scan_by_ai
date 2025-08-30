// Modern Home Page Functionality
document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const fileInput = document.getElementById('file-input');
    const uploadContainer = document.getElementById('upload-container');
    const uploadPrompt = document.getElementById('upload-prompt');
    const imagePreview = document.getElementById('image-preview');
    const previewImage = document.getElementById('preview-image');
    const loadingState = document.getElementById('loading-state');
    const resultsSection = document.getElementById('results-section');
    const analyzeBtn = document.getElementById('analyze-btn');
    const removeBtn = document.getElementById('remove-btn');
    const analyzeAnotherBtn = document.getElementById('analyze-another-btn');
    const downloadReportBtn = document.getElementById('download-report-btn');
    
    // Modal elements
    const analysisModal = document.getElementById('analysis-modal');
    const reportModal = document.getElementById('report-modal');
    const closeReportModal = document.getElementById('close-report-modal');
    const generateReportBtn = document.getElementById('generate-report-btn');
    const cancelReport = document.getElementById('cancel-report');
    
    let selectedFile = null;
    let analysisResult = null;
    let currentImageUrl = null;
    
    // File input handling
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleFileSelection(file);
        }
    });
    
    // Drag and drop handling
    uploadContainer.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadContainer.style.borderColor = 'var(--primary-400)';
        uploadContainer.style.background = 'rgba(59, 130, 246, 0.05)';
    });
    
    uploadContainer.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadContainer.style.borderColor = '';
        uploadContainer.style.background = '';
    });
    
    uploadContainer.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadContainer.style.borderColor = '';
        uploadContainer.style.background = '';
        
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            handleFileSelection(file);
        } else {
            showToast('Please select a valid image file', 'error');
        }
    });
    
    // Button event listeners
    analyzeBtn?.addEventListener('click', analyzeImage);
    removeBtn?.addEventListener('click', removeImage);
    analyzeAnotherBtn?.addEventListener('click', resetForNewAnalysis);
    downloadReportBtn?.addEventListener('click', showReportModal);
    
    // Modal event listeners
    closeReportModal?.addEventListener('click', hideReportModal);
    cancelReport?.addEventListener('click', hideReportModal);
    generateReportBtn?.addEventListener('click', generatePDFReport);
    
    function handleFileSelection(file) {
        // Validate file type
        const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp', 'image/webp'];
        if (!allowedTypes.includes(file.type)) {
            showToast('Please select a valid image file (PNG, JPG, JPEG, GIF, BMP, WebP)', 'error');
            return;
        }
        
        // Validate file size (16MB limit)
        if (file.size > 16 * 1024 * 1024) {
            showToast('File size must be less than 16MB', 'error');
            return;
        }
        
        selectedFile = file;
        
        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            currentImageUrl = e.target.result;
            showImagePreview();
        };
        reader.readAsDataURL(file);
    }
    
    function showImagePreview() {
        uploadPrompt.classList.add('hidden');
        imagePreview.classList.remove('hidden');
        resultsSection.classList.add('hidden');
        
        // Add smooth transition
        imagePreview.classList.add('animate-scaleIn');
        
        lucide.createIcons();
        showToast('Image loaded successfully! Ready for analysis.', 'success');
    }
    
    function removeImage() {
        selectedFile = null;
        currentImageUrl = null;
        analysisResult = null;
        fileInput.value = '';
        
        uploadPrompt.classList.remove('hidden');
        imagePreview.classList.add('hidden');
        resultsSection.classList.add('hidden');
        loadingState.classList.add('hidden');
        
        showToast('Image removed', 'info');
    }
    
    function resetForNewAnalysis() {
        removeImage();
        showToast('Ready for a new analysis! Please upload an image.', 'info');
    }
    
    async function analyzeImage() {
        if (!selectedFile) {
            showToast('Please upload a new image first', 'error');
            return;
        }
        
        // Check if analysis was already performed
        if (analysisResult) {
            showToast('Please upload a new image for analysis', 'info');
            resetForNewAnalysis();
            return;
        }
        
        // Show analysis modal
        showAnalysisModal();
        
        // Start countdown
        let countdown = 5;
        const countdownElement = document.getElementById('countdown');
        const countdownInterval = setInterval(() => {
            countdown--;
            if (countdownElement) {
                countdownElement.textContent = countdown;
            }
            
            if (countdown <= 0) {
                clearInterval(countdownInterval);
                hideAnalysisModal();
                performAnalysis();
            }
        }, 1000);
    }
    
    async function performAnalysis() {
        // Show loading state
        imagePreview.classList.add('hidden');
        loadingState.classList.remove('hidden');
        loadingState.classList.add('animate-fadeIn');
        
        try {
            const formData = new FormData();
            formData.append('image', selectedFile);
            
            const response = await fetch('/api/analyze', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Analysis failed');
            }
            
            analysisResult = await response.json();
            
            // Save to local storage
            saveAnalysisToLocalStorage(analysisResult, currentImageUrl);
            
            // Show results
            displayResults(analysisResult);
            showToast('Analysis completed successfully!', 'success');
            
        } catch (error) {
            console.error('Analysis error:', error);
            showToast(error.message || 'Failed to analyze image', 'error');
            
            // Return to preview state
            loadingState.classList.add('hidden');
            imagePreview.classList.remove('hidden');
        }
    }
    
    function displayResults(result) {
        // Hide loading state
        loadingState.classList.add('hidden');
        
        // Clear the selected file after analysis to require new upload
        selectedFile = null;
        fileInput.value = '';
        
        // Determine if image is AI-generated first
        const isAIGenerated = result.classification.toLowerCase().includes('ai') || 
                             result.classification.toLowerCase().includes('generated') ||
                             result.classification.toLowerCase().includes('artificial');
        
        // Display the analyzed image in results with professional circular styling
        const resultsImage = document.getElementById('results-image');
        if (currentImageUrl) {
            resultsImage.src = currentImageUrl;
            // Add professional circular border and wave effects based on classification
            const imageContainer = document.getElementById('results-image-container');
            if (imageContainer) {
                if (isAIGenerated) {
                    imageContainer.className = 'aspect-square rounded-full overflow-hidden bg-gray-100 dark:bg-gray-800 image-ai-generated ai-detected-wave';
                } else {
                    imageContainer.className = 'aspect-square rounded-full overflow-hidden bg-gray-100 dark:bg-gray-800 image-real real-detected-wave';
                }
            }
        }
        
        // Update image details
        const fileName = document.getElementById('file-name');
        const imageSize = document.getElementById('image-size');
        
        if (fileName) fileName.textContent = result.filename;
        if (imageSize) imageSize.textContent = result.imageSize;
        
        // Update result badge
        const resultBadge = document.getElementById('result-badge');
        const resultIcon = document.getElementById('result-icon');
        const resultText = document.getElementById('result-text');
        
        if (isAIGenerated) {
            resultBadge.className = 'badge badge-ai-generated text-xl px-8 py-4 inline-flex items-center gap-3 rounded-2xl';
            resultIcon.setAttribute('data-lucide', 'alert-triangle');
            resultIcon.className = 'w-8 h-8';
            resultText.textContent = result.classification;
        } else {
            resultBadge.className = 'badge badge-real text-xl px-8 py-4 inline-flex items-center gap-3 rounded-2xl';
            resultIcon.setAttribute('data-lucide', 'shield-check');
            resultIcon.className = 'w-8 h-8';
            resultText.textContent = result.classification;
        }
        
        // Update all section header colors based on classification
        updateSectionHeaderColors(isAIGenerated);
        
        // 🔴 ULTRA FORCE BUTTON COLORS - MAKE THEM RED FOR AI GENERATED!
        console.log('🚨 ULTRA FORCE: AI Generated =', isAIGenerated, 'Classification:', result.classification);
        
        // Get ALL buttons using multiple methods
        const allButtons = [
            ...document.querySelectorAll('button'),
            ...document.querySelectorAll('[role="button"]'),
            ...document.querySelectorAll('input[type="button"]'),
            ...document.querySelectorAll('a[class*="btn"]')
        ];
        
        console.log('🎯 Found', allButtons.length, 'potential buttons');
        
        // Target specific buttons by ID and text content
        const targetButtons = [
            document.getElementById('analyze-btn'),
            document.getElementById('analyze-another-btn'),
            document.getElementById('download-report-btn'),
            ...allButtons.filter(btn => {
                const text = (btn.textContent || '').toLowerCase();
                return text.includes('analyz') || 
                       text.includes('download') || 
                       text.includes('report') || 
                       text.includes('new') ||
                       btn.id.includes('analyz') ||
                       btn.id.includes('report');
            })
        ].filter(btn => btn); // Remove nulls
        
        console.log('🎯 Target buttons found:', targetButtons.length);
        
        targetButtons.forEach((button, i) => {
            console.log(`Button ${i}:`, button.textContent, button.id, button.className);
            
            if (isAIGenerated) {
                console.log('🔴 FORCE RED FOR AI GENERATED');
                // FORCE RED COLORS FOR AI GENERATED
                const redStyle = `
                    background: #dc2626 !important;
                    background-color: #dc2626 !important;
                    background-image: linear-gradient(135deg, #ef4444, #dc2626) !important;
                    color: white !important;
                    border: 2px solid #b91c1c !important;
                    padding: 0.75rem 1.5rem !important;
                    border-radius: 0.75rem !important;
                    font-weight: 600 !important;
                    transition: all 0.3s ease !important;
                    display: inline-flex !important;
                    align-items: center !important;
                    justify-content: center !important;
                    gap: 0.5rem !important;
                `;
                
                // Apply styles multiple ways to ensure it works
                button.style.cssText = redStyle;
                button.setAttribute('style', redStyle);
                button.style.setProperty('background-color', '#dc2626', 'important');
                button.style.setProperty('color', 'white', 'important');
                
                // Remove conflicting classes
                button.classList.remove('bg-gradient-to-r', 'from-primary-500', 'to-secondary-500');
                button.classList.remove('bg-gray-100', 'dark:bg-gray-700', 'text-gray-700');
                
            } else {
                console.log('🟢 FORCE GREEN FOR REAL IMAGE');
                // FORCE GREEN COLORS FOR REAL IMAGE
                const greenStyle = `
                    background: #059669 !important;
                    background-color: #059669 !important;
                    background-image: linear-gradient(135deg, #10b981, #059669) !important;
                    color: white !important;
                    border: 2px solid #047857 !important;
                    padding: 0.75rem 1.5rem !important;
                    border-radius: 0.75rem !important;
                    font-weight: 600 !important;
                    transition: all 0.3s ease !important;
                    display: inline-flex !important;
                    align-items: center !important;
                    justify-content: center !important;
                    gap: 0.5rem !important;
                `;
                
                // Apply styles multiple ways to ensure it works
                button.style.cssText = greenStyle;
                button.setAttribute('style', greenStyle);
                button.style.setProperty('background-color', '#059669', 'important');
                button.style.setProperty('color', 'white', 'important');
                
                // Remove conflicting classes
                button.classList.remove('bg-gradient-to-r', 'from-primary-500', 'to-secondary-500');
                button.classList.remove('bg-gray-100', 'dark:bg-gray-700', 'text-gray-700');
            }
            
            // Force repaint
            button.offsetHeight;
            setTimeout(() => button.offsetHeight, 100);
        });
        
        // 🎯 FORCE TEXT AND ICON COLORS INSIDE BUTTONS
        targetButtons.forEach(button => {
            // Target all text elements inside buttons
            const textElements = button.querySelectorAll('span, div, p');
            const iconElements = button.querySelectorAll('i, svg');
            
            textElements.forEach(text => {
                if (isAIGenerated) {
                    text.style.setProperty('color', 'white', 'important');
                    text.style.setProperty('text-color', 'white', 'important');
                } else {
                    text.style.setProperty('color', 'white', 'important');
                    text.style.setProperty('text-color', 'white', 'important');
                }
            });
            
            iconElements.forEach(icon => {
                if (isAIGenerated) {
                    icon.style.setProperty('color', 'white', 'important');
                    icon.style.setProperty('fill', 'white', 'important');
                    icon.style.setProperty('stroke', 'white', 'important');
                } else {
                    icon.style.setProperty('color', 'white', 'important');
                    icon.style.setProperty('fill', 'white', 'important');
                    icon.style.setProperty('stroke', 'white', 'important');
                }
            });
        });
        
        // Additional timeout to ensure styles stick
        setTimeout(() => {
            targetButtons.forEach(button => {
                if (isAIGenerated) {
                    button.style.setProperty('background-color', '#dc2626', 'important');
                    button.style.setProperty('color', 'white', 'important');
                    
                    // Re-apply to inner elements
                    const allInnerElements = button.querySelectorAll('*');
                    allInnerElements.forEach(element => {
                        element.style.setProperty('color', 'white', 'important');
                        element.style.setProperty('fill', 'white', 'important');
                        element.style.setProperty('stroke', 'white', 'important');
                    });
                } else {
                    button.style.setProperty('background-color', '#059669', 'important');
                    button.style.setProperty('color', 'white', 'important');
                    
                    // Re-apply to inner elements
                    const allInnerElements = button.querySelectorAll('*');
                    allInnerElements.forEach(element => {
                        element.style.setProperty('color', 'white', 'important');
                        element.style.setProperty('fill', 'white', 'important');
                        element.style.setProperty('stroke', 'white', 'important');
                    });
                }
            });
        }, 500);
        
        // Update confidence score with professional color styling
        const confidencePercentage = document.getElementById('confidence-percentage');
        const confidenceBar = document.getElementById('confidence-bar');
        
        if (confidencePercentage) {
            confidencePercentage.textContent = `${result.confidence}%`;
            // Apply professional red or green styling to the percentage text
            if (isAIGenerated) {
                confidencePercentage.className = 'text-2xl font-bold text-red-600 dark:text-red-400';
                confidencePercentage.style.color = '#dc2626';
                confidencePercentage.style.textShadow = '0 2px 4px rgba(220, 38, 38, 0.2)';
                confidencePercentage.style.fontWeight = '700';
            } else {
                confidencePercentage.className = 'text-2xl font-bold text-green-600 dark:text-green-400';
                confidencePercentage.style.color = '#7ADAA5';
                confidencePercentage.style.textShadow = '0 2px 4px rgba(122, 218, 165, 0.2)';
                confidencePercentage.style.fontWeight = '700';
            }
        }
        
        if (confidenceBar) {
            // Clear any existing styles and classes first
            confidenceBar.className = '';
            confidenceBar.style.cssText = '';
            
            // Set width with animation
            setTimeout(() => {
                confidenceBar.style.width = `${result.confidence}%`;
            }, 100);
            
            // Apply colors based on classification with strong inline styles
            if (isAIGenerated) {
                confidenceBar.className = 'progress-bar-ai-generated h-4 rounded-full';
                confidenceBar.style.cssText = `
                    width: ${result.confidence}%;
                    height: 16px;
                    background: linear-gradient(90deg, #dc2626 0%, #b91c1c 50%, #991b1b 100%) !important;
                    border: 2px solid rgba(220, 38, 38, 0.5) !important;
                    border-radius: 9999px !important;
                    transition: all 1000ms ease-out !important;
                    position: relative !important;
                `;
            } else {
                confidenceBar.className = 'progress-bar-real h-4 rounded-full';
                confidenceBar.style.cssText = `
                    width: ${result.confidence}%;
                    height: 16px;
                    background: linear-gradient(90deg, #7ADAA5 0%, #5fb88b 50%, #4a9a72 100%) !important;
                    border: 2px solid rgba(122, 218, 165, 0.5) !important;
                    border-radius: 9999px !important;
                    transition: all 1000ms ease-out !important;
                    position: relative !important;
                `;
            }
            
            // Force a repaint
            confidenceBar.offsetHeight;
        }
        
        // Update processing time
        const processingTime = document.getElementById('processing-time');
        if (processingTime) processingTime.textContent = `${result.processingTime}s`;
        
        // Update detection indicators
        const indicatorsGrid = document.getElementById('indicators-grid');
        if (indicatorsGrid) {
            indicatorsGrid.innerHTML = '';
            
            if (result.indicators && result.indicators.length > 0) {
                result.indicators.forEach(indicator => {
                    const indicatorElement = document.createElement('div');
                    indicatorElement.className = 'result-card p-4 flex justify-between items-center hover-lift';
                    
                    const leftDiv = document.createElement('div');
                    leftDiv.className = 'flex items-center gap-3';
                    
                    const icon = document.createElement('i');
                    icon.setAttribute('data-lucide', getIndicatorIcon(indicator.strength));
                    icon.className = isAIGenerated ? 'w-5 h-5 text-red-500' : 'w-5 h-5 text-green-500';
                    
                    const nameSpan = document.createElement('span');
                    nameSpan.className = 'font-medium';
                    nameSpan.textContent = indicator.name;
                    
                    const strengthBadge = document.createElement('span');
                    strengthBadge.className = getStrengthBadgeClass(indicator.strength);
                    strengthBadge.textContent = indicator.strength;
                    
                    leftDiv.appendChild(icon);
                    leftDiv.appendChild(nameSpan);
                    indicatorElement.appendChild(leftDiv);
                    indicatorElement.appendChild(strengthBadge);
                    indicatorsGrid.appendChild(indicatorElement);
                });
            } else {
                const noIndicators = document.createElement('div');
                noIndicators.className = 'text-center py-8 text-gray-500 dark:text-gray-400 col-span-2';
                noIndicators.innerHTML = `
                    <i data-lucide="info" class="w-8 h-8 mx-auto mb-3 opacity-50"></i>
                    <p>No specific detection indicators available</p>
                `;
                indicatorsGrid.appendChild(noIndicators);
            }
        }
        
        // Show results section with animation
        resultsSection.classList.remove('hidden');
        resultsSection.classList.add('animate-fadeIn');
        lucide.createIcons();
        
        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
    
    function getStrengthBadgeClass(strength) {
        switch (strength.toLowerCase()) {
            case 'strong': return 'indicator-strong';
            case 'moderate': return 'indicator-moderate';
            case 'weak': return 'indicator-weak';
            default: return 'badge text-xs px-3 py-1 bg-gray-100 text-gray-700 border border-gray-200';
        }
    }
    
    function showAnalysisModal() {
        if (analysisModal) {
            analysisModal.classList.remove('hidden');
            document.body.style.overflow = 'hidden';
        }
    }
    
    function hideAnalysisModal() {
        if (analysisModal) {
            analysisModal.classList.add('hidden');
            document.body.style.overflow = 'unset';
        }
    }
    
    function showReportModal() {
        if (reportModal) {
            reportModal.classList.remove('hidden');
            document.body.style.overflow = 'hidden';
        }
    }
    
    function hideReportModal() {
        if (reportModal) {
            reportModal.classList.add('hidden');
            document.body.style.overflow = 'unset';
        }
    }
    
    function generatePDFReport() {
        if (!analysisResult || !currentImageUrl) {
            showToast('No analysis data available', 'error');
            return;
        }
        
        hideReportModal();
        
        const isAIGenerated = analysisResult.classification.toLowerCase().includes('ai') || 
                             analysisResult.classification.toLowerCase().includes('generated') ||
                             analysisResult.classification.toLowerCase().includes('artificial');
        
        // Define styles based on classification
        const badgeStyles = isAIGenerated ? 
            'background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); color: white; border: 2px solid #fca5a5;' : 
            'background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; border: 2px solid #86efac;';
        
        const badgeIcon = isAIGenerated ? '⚠️' : '✅';
        
        const confidenceBarStyles = isAIGenerated ? 
            'background: linear-gradient(90deg, #ef4444 0%, #dc2626 100%);' : 
            'background: linear-gradient(90deg, #10b981 0%, #059669 100%);';
        
        // Create HTML content for the report
        const reportHTML = `
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>AI Detection Report</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        @page { size: A4; margin: 15mm; }
        body { 
            font-family: 'Inter', 'Segoe UI', sans-serif; 
            line-height: 1.4; 
            color: #374151; 
            background: #ffffff;
            font-size: 12px;
        }
        .header {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
            color: white;
            padding: 20px;
            text-align: center;
            margin-bottom: 20px;
            border-radius: 12px;
        }
        .header h1 { font-size: 2em; margin-bottom: 8px; font-weight: 700; }
        .header p { font-size: 1em; opacity: 0.9; }
        .top-section { 
            display: flex; 
            gap: 20px; 
            margin-bottom: 20px; 
            align-items: flex-start;
        }
        .image-container { 
            flex: 1; 
            text-align: center;
            background: #f9fafb;
            border: 2px solid #e5e7eb;
            border-radius: 12px;
            padding: 15px;
        }
        .analyzed-image { 
            max-width: 100%; 
            max-height: 200px;
            border-radius: 8px; 
            margin-bottom: 10px;
        }
        .classification-container { flex: 1; }
        .classification-badge {
            padding: 20px 30px;
            border-radius: 50px;
            text-align: center;
            font-weight: 700;
            font-size: 1.3em;
            margin-bottom: 20px;
            display: inline-flex;
            align-items: center;
            gap: 12px;
            ${badgeStyles}
        }
        .classification-badge::before {
            content: '${badgeIcon}';
            font-size: 1.2em;
        }
        .confidence-section {
            background: #f9fafb;
            border: 2px solid #e5e7eb;
            border-radius: 12px;
            padding: 15px;
        }
        .confidence-bar-container {
            background: #e5e7eb;
            height: 15px;
            border-radius: 8px;
            overflow: hidden;
            margin: 10px 0;
        }
        .confidence-bar {
            height: 100%;
            width: ${analysisResult.confidence}%;
            ${confidenceBarStyles}
        }
        .section {
            background: #f9fafb;
            border: 2px solid #e5e7eb;
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 15px;
        }
        .section-title {
            font-weight: 700;
            font-size: 1.1em;
            margin-bottom: 10px;
            color: #1f2937;
        }
        .indicator-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 12px;
            margin-bottom: 8px;
            border-radius: 8px;
            background: white;
            border: 1px solid #e5e7eb;
        }
        .footer {
            background: #f3f4f6;
            border: 2px solid #d1d5db;
            border-radius: 12px;
            padding: 15px;
            margin-top: 20px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>AI Detection Analysis Report</h1>
        <p>Professional Image Authentication Analysis • ${new Date().toLocaleDateString()}</p>
    </div>

    <div class="top-section">
        <div class="image-container">
            <h3>Analyzed Image</h3>
            <img src="${currentImageUrl}" alt="Analyzed Image" class="analyzed-image">
            <div style="margin-top: 10px; font-size: 0.9em; color: #6b7280;">
                ${analysisResult.filename}
            </div>
        </div>
        
        <div class="classification-container">
            <div class="classification-badge">
                ${analysisResult.classification}
            </div>
            
            <div class="confidence-section">
                <div style="font-weight: 600; margin-bottom: 5px;">Confidence Score</div>
                <div class="confidence-bar-container">
                    <div class="confidence-bar"></div>
                </div>
                <div style="text-align: center; font-weight: 600; font-size: 1.1em;">
                    ${analysisResult.confidence}%
                </div>
            </div>
        </div>
    </div>

    <div class="section">
        <div class="section-title">Detection Indicators</div>
        ${analysisResult.indicators && analysisResult.indicators.length > 0 ? 
            analysisResult.indicators.map(indicator => `
                <div class="indicator-item">
                    <span>${indicator.name}</span>
                    <span style="font-weight: 600;">${indicator.strength}</span>
                </div>
            `).join('') : 
            '<div style="text-align: center; color: #6b7280; padding: 20px;">No specific indicators detected</div>'
        }
    </div>

    <div class="section">
        <div class="section-title">Technical Details</div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
            <div>File Name: <strong>${analysisResult.filename}</strong></div>
            <div>Image Size: <strong>${analysisResult.imageSize}</strong></div>
            <div>Processing Time: <strong>${analysisResult.processingTime}s</strong></div>
            <div>Method: <strong>${analysisResult.methodUsed || 'Computer Vision'}</strong></div>
        </div>
    </div>

    <div class="footer">
        <div style="font-weight: 600; margin-bottom: 5px;">AI Image Detection Service</div>
        <div style="font-size: 0.9em; color: #6b7280;">
            This report was generated using advanced machine learning algorithms for image authenticity verification.
            Results are provided for informational purposes and should be considered alongside other verification methods.
        </div>
    </div>
</body>
</html>`;
        
        // Create and open new window for printing
        const printWindow = window.open('', '_blank');
        printWindow.document.write(reportHTML);
        printWindow.document.close();
        
        // Wait for content to load then trigger print
        printWindow.onload = () => {
            setTimeout(() => {
                printWindow.print();
                printWindow.close();
            }, 1000);
        };
        
        showToast('Report generated successfully! Print dialog opened.', 'success');
    }
});

// Enhanced Local Storage Management
function saveAnalysisToLocalStorage(analysis, imageData) {
    try {
        const analysisHistory = JSON.parse(localStorage.getItem('analysisHistory') || '[]');
        const newAnalysis = {
            ...analysis,
            timestamp: new Date().toISOString(),
            imageData: imageData,
            id: Date.now().toString()
        };
        
        analysisHistory.unshift(newAnalysis);
        
        // Keep only last 20 analyses
        if (analysisHistory.length > 20) {
            analysisHistory.splice(20);
        }
        
        localStorage.setItem('analysisHistory', JSON.stringify(analysisHistory));
    } catch (error) {
        console.error('Error saving to local storage:', error);
        showToast('Failed to save analysis history', 'warning');
    }
}

// Utility function for indicator strength badge colors
function getStrengthBadgeColor(strength) {
    switch (strength.toLowerCase()) {
        case 'strong': return 'indicator-strong';
        case 'moderate': return 'indicator-moderate';
        case 'weak': return 'indicator-weak';
        default: return 'bg-gray-100 text-gray-700 border border-gray-200';
    }
}

// Update section header colors based on AI detection result
function updateSectionHeaderColors(isAIGenerated) {
    // Define color classes based on result with specific green color #7ADAA5
    const iconColorClass = isAIGenerated ? 'text-red-500' : 'text-green-500';
    const iconStyleColor = isAIGenerated ? '#dc2626' : '#7ADAA5';
    const headerBgClass = isAIGenerated ? 
        'bg-gradient-to-br from-red-500 to-red-600' : 
        'bg-gradient-to-br from-green-500 to-green-600';
    
    // Update "Analyzed Image" section header
    const analyzedImageHeader = document.querySelector('h3:has([data-lucide="image"])');
    if (analyzedImageHeader) {
        const iconContainer = analyzedImageHeader.querySelector('div');
        const icon = analyzedImageHeader.querySelector('i');
        if (iconContainer) iconContainer.className = `w-8 h-8 ${headerBgClass} rounded-lg flex items-center justify-center mr-3`;
        if (icon) icon.className = 'w-5 h-5 text-white';
    }
    
    // Update "Detection Results" section header  
    const detectionHeader = document.querySelector('h3:has([data-lucide="brain"])');
    if (detectionHeader) {
        const iconContainer = detectionHeader.querySelector('div');
        const icon = detectionHeader.querySelector('i');
        if (iconContainer) iconContainer.className = `w-8 h-8 ${headerBgClass} rounded-lg flex items-center justify-center mr-3`;
        if (icon) icon.className = 'w-5 h-5 text-white';
    }
    
    // Update "Image Details" section header
    const imageDetailsHeader = document.querySelector('h4:has([data-lucide="info"])');
    if (imageDetailsHeader) {
        const icon = imageDetailsHeader.querySelector('i');
        if (icon) icon.className = `w-5 h-5 ${iconColorClass}`;
    }
    
    // Update "Confidence Score" icon and label color
    const confidenceIcon = document.querySelector('[data-lucide="bar-chart-3"]');
    if (confidenceIcon) {
        confidenceIcon.className = `w-5 h-5 ${iconColorClass}`;
        confidenceIcon.style.color = iconStyleColor;
    }
    
    // Update confidence score section label color
    const confidenceHeaders = document.querySelectorAll('h4');
    confidenceHeaders.forEach(header => {
        if (header.textContent.includes('Confidence Score')) {
            header.style.color = iconStyleColor;
        }
    });
    
    // Update "Detection Indicators" section header
    const indicatorsHeader = document.querySelector('h4:has([data-lucide="eye"])');
    if (indicatorsHeader) {
        const iconContainer = indicatorsHeader.querySelector('div');
        const icon = indicatorsHeader.querySelector('i');
        if (iconContainer) iconContainer.className = `w-8 h-8 ${headerBgClass} rounded-lg flex items-center justify-center mr-3`;
        if (icon) icon.className = 'w-5 h-5 text-white';
    }
}

// Enhanced indicator icon based on result type
function getIndicatorIcon(strength) {
    switch (strength.toLowerCase()) {
        case 'strong': return 'alert-triangle';
        case 'moderate': return 'alert-circle';
        case 'weak': return 'info';
        default: return 'help-circle';
    }
}