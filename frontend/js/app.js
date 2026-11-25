/**
 * Skin Disease Detection — Frontend Application Logic
 * Modular controller managing image acquisition, quality checking, prediction inference,
 * on-page clinical reports, disease library explorer, and persistent reports management.
 */

let selectedFile = null;
let cachedLibraryData = [];
let cachedReports = [];

// Track active report IDs for non-reloading asynchronous PDF downloads
let currentAnalysisRecordId = null;
let currentAnalysisRefId = null;
let currentModalReportId = null;
let currentModalReportRef = null;
const activeDownloads = new Set();

/**
 * Asynchronous, Non-Reloading PDF Downloader
 * Converts PDF stream into a Blob and triggers direct anchor download.
 * Preserves current application state, loaded image, charts, and scroll position.
 */
async function downloadReportPdfAsync(recordId, refId, triggerBtn) {
    if (!recordId) {
        showAlert("No valid report ID available for download.", "warning");
        return;
    }

    const downloadKey = String(recordId);
    if (activeDownloads.has(downloadKey)) {
        return;
    }
    activeDownloads.add(downloadKey);

    let originalHtml = "";
    if (triggerBtn) {
        triggerBtn.disabled = true;
        originalHtml = triggerBtn.innerHTML;
        triggerBtn.innerHTML = `
            <svg class="btn-spinner" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" style="width: 14px; height: 14px; animation: spin 0.7s linear infinite; vertical-align: middle; margin-right: 6px;">
                <circle cx="12" cy="12" r="10" stroke-opacity="0.25"></circle>
                <path d="M12 2a10 10 0 0 1 10 10" stroke-linecap="round"></path>
            </svg>
            <span>Preparing PDF...</span>
        `;
    }

    try {
        const endpoint = `/api/report/${recordId}`;
        const response = await fetch(endpoint, {
            method: "GET",
            headers: { "Accept": "application/pdf" }
        });

        if (!response.ok) {
            let errorMsg = `Server error (HTTP ${response.status})`;
            try {
                const errJson = await response.json();
                if (errJson && errJson.detail) errorMsg = errJson.detail;
            } catch (_) {}
            throw new Error(errorMsg);
        }

        const contentType = response.headers.get("content-type") || "";
        if (!contentType.includes("pdf") && !contentType.includes("octet-stream")) {
            throw new Error("Invalid document format received from server.");
        }

        const blob = await response.blob();
        if (blob.size === 0) {
            throw new Error("Received empty PDF document from server.");
        }

        const safeRef = refId ? String(refId).replace(/[^a-zA-Z0-9_-]/g, "") : `REC${recordId}`;
        const downloadFilename = `Report_${safeRef}.pdf`;

        // Check for legacy navigator.msSaveOrOpenBlob
        if (window.navigator && window.navigator.msSaveOrOpenBlob) {
            window.navigator.msSaveOrOpenBlob(blob, downloadFilename);
            return;
        }

        // Standard modern blob download via temporary download anchor
        const blobUrl = URL.createObjectURL(blob);
        const downloadAnchor = document.createElement("a");
        downloadAnchor.style.display = "none";
        downloadAnchor.href = blobUrl;
        downloadAnchor.download = downloadFilename;
        downloadAnchor.target = "_self";
        document.body.appendChild(downloadAnchor);
        downloadAnchor.click();

        setTimeout(() => {
            if (downloadAnchor.parentNode) {
                downloadAnchor.parentNode.removeChild(downloadAnchor);
            }
            URL.revokeObjectURL(blobUrl);
        }, 2000);

    } catch (err) {
        console.error("PDF download failed:", err);
        const errorAlert = document.getElementById("alert-container");
        if (errorAlert) {
            showAlert(`PDF download failed: ${err.message}`, "danger");
        } else {
            alert(`PDF download failed: ${err.message}`);
        }
    } finally {
        activeDownloads.delete(downloadKey);
        if (triggerBtn) {
            triggerBtn.disabled = false;
            triggerBtn.innerHTML = originalHtml;
        }
    }
}

document.addEventListener("DOMContentLoaded", () => {
    initNavigation();
    initSkinAnalysis();
    initDiseaseLibrary();
    initReportsManagement();
    initLightbox();
});

/* ==========================================================================
   1. Header Navigation & Mobile Sliding Off-Canvas Drawer
   ========================================================================== */
function initNavigation() {
    const hamburgerBtn = document.getElementById("btn-hamburger");
    const closeDrawerBtn = document.getElementById("btn-close-drawer");
    const mobileDrawer = document.getElementById("mobile-drawer");
    const drawerOverlay = document.getElementById("drawer-overlay");
    const drawerLinks = document.querySelectorAll(".drawer-item-link");

    function openDrawer() {
        if (!mobileDrawer) return;
        mobileDrawer.classList.add("open");
        mobileDrawer.setAttribute("aria-hidden", "false");
        if (hamburgerBtn) hamburgerBtn.setAttribute("aria-expanded", "true");
        if (drawerOverlay) {
            drawerOverlay.classList.add("active");
            drawerOverlay.setAttribute("aria-hidden", "false");
        }
        document.body.classList.add("drawer-open");
    }

    function closeDrawer() {
        if (!mobileDrawer) return;
        mobileDrawer.classList.remove("open");
        mobileDrawer.setAttribute("aria-hidden", "true");
        if (hamburgerBtn) hamburgerBtn.setAttribute("aria-expanded", "false");
        if (drawerOverlay) {
            drawerOverlay.classList.remove("active");
            drawerOverlay.setAttribute("aria-hidden", "true");
        }
        document.body.classList.remove("drawer-open");
    }

    if (hamburgerBtn) {
        hamburgerBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            if (mobileDrawer && mobileDrawer.classList.contains("open")) {
                closeDrawer();
            } else {
                openDrawer();
            }
        });
    }

    if (closeDrawerBtn) {
        closeDrawerBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            closeDrawer();
        });
    }

    if (drawerOverlay) {
        drawerOverlay.addEventListener("click", () => {
            closeDrawer();
        });
    }

    drawerLinks.forEach((link) => {
        link.addEventListener("click", () => {
            closeDrawer();
        });
    });

    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && mobileDrawer && mobileDrawer.classList.contains("open")) {
            closeDrawer();
        }
    });

    window.addEventListener("resize", () => {
        if (window.innerWidth > 768 && mobileDrawer && mobileDrawer.classList.contains("open")) {
            closeDrawer();
        }
    });
}

/* ==========================================================================
   2. Skin Image Analysis (Upload, Mobile Camera, Quality Check & Predict)
   ========================================================================== */
function initSkinAnalysis() {
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("file-input");
    const cameraInput = document.getElementById("camera-input");
    const btnChooseFile = document.getElementById("btn-choose-file");
    const btnTakePhoto = document.getElementById("btn-take-photo");
    const previewBox = document.getElementById("preview-box");
    const previewImg = document.getElementById("preview-img");
    const previewFilename = document.getElementById("preview-filename");
    const previewFilesize = document.getElementById("preview-filesize");
    const btnRetake = document.getElementById("btn-retake");
    const btnRemove = document.getElementById("btn-remove-image");
    const btnAnalyze = document.getElementById("btn-analyze");
    const btnNewAnalysis = document.getElementById("btn-new-analysis");
    const btnPrintReport = document.getElementById("btn-print-report");
    const btnNewAnalysisBottom = document.getElementById("btn-new-analysis-bottom");
    const btnPrintReportBottom = document.getElementById("btn-print-report-bottom");
    const loadingSpinner = document.getElementById("loading-spinner");
    const reportSection = document.getElementById("report-section");

    // Camera Modal Elements
    const cameraModal = document.getElementById("camera-modal");
    const cameraVideo = document.getElementById("camera-video");
    const cameraPreviewImg = document.getElementById("camera-preview-img");
    const cameraCanvas = document.getElementById("camera-canvas");
    const cameraAlert = document.getElementById("camera-alert");
    const cameraAlertText = document.getElementById("camera-alert-text");
    const btnCameraFallback = document.getElementById("btn-camera-fallback");
    const btnSwitchCamera = document.getElementById("btn-switch-camera");
    const cameraReticle = document.getElementById("camera-reticle");
    const cameraLiveControls = document.getElementById("camera-live-controls");
    const cameraReviewControls = document.getElementById("camera-review-controls");
    const btnCameraCapture = document.getElementById("btn-camera-capture");
    const btnCameraCancel = document.getElementById("btn-camera-cancel");
    const btnCameraRetake = document.getElementById("btn-camera-retake");
    const btnCameraUse = document.getElementById("btn-camera-use");
    const btnCloseCamera = document.getElementById("btn-close-camera");
    const cameraViewport = document.getElementById("camera-viewport");

    // Camera & Upload state
    let lastInputSource = "file";
    let cameraStream = null;
    let videoDevices = [];
    let currentDeviceIndex = 0;
    let capturedBlob = null;
    let currentFacingMode = "environment";

    if (!dropzone) return;

    // Trigger file chooser
    if (btnChooseFile && fileInput) {
        btnChooseFile.addEventListener("click", (e) => {
            e.stopPropagation();
            fileInput.click();
        });
    }

    // Trigger live camera modal
    if (btnTakePhoto) {
        btnTakePhoto.addEventListener("click", (e) => {
            e.stopPropagation();
            openCameraModal();
        });
    }

    // Drag-and-drop support
    ["dragenter", "dragover"].forEach(evt => {
        dropzone.addEventListener(evt, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add("dragover");
        });
    });

    ["dragleave", "drop"].forEach(evt => {
        dropzone.addEventListener(evt, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove("dragover");
        });
    });

    dropzone.addEventListener("drop", (e) => {
        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            lastInputSource = "file";
            handleFileSelection(files[0]);
        }
    });

    // File input changes
    if (fileInput) {
        fileInput.addEventListener("change", (e) => {
            if (e.target.files && e.target.files.length > 0) {
                lastInputSource = "file";
                handleFileSelection(e.target.files[0]);
            }
        });
    }

    // Camera input changes (fallback)
    if (cameraInput) {
        cameraInput.addEventListener("change", (e) => {
            if (e.target.files && e.target.files.length > 0) {
                lastInputSource = "camera";
                handleFileSelection(e.target.files[0]);
            }
        });
    }

    // Retake / Change
    if (btnRetake) {
        btnRetake.addEventListener("click", () => {
            if (lastInputSource === "camera") {
                openCameraModal();
            } else if (fileInput) {
                fileInput.click();
            }
        });
    }

    // Remove Image
    if (btnRemove) {
        btnRemove.addEventListener("click", () => {
            resetUploadState();
        });
    }

    // Start New Analysis (Top & Bottom)
    function handleNewAnalysisClick() {
        resetUploadState();
        if (reportSection) reportSection.style.display = "none";
        clearAlert();
        window.scrollTo({ top: 0, behavior: "smooth" });
    }

    if (btnNewAnalysis) {
        btnNewAnalysis.addEventListener("click", handleNewAnalysisClick);
    }
    if (btnNewAnalysisBottom) {
        btnNewAnalysisBottom.addEventListener("click", handleNewAnalysisClick);
    }

    // Print Report (Top & Bottom)
    function handlePrintReportClick() {
        window.print();
    }

    if (btnPrintReport) {
        btnPrintReport.addEventListener("click", handlePrintReportClick);
    }
    if (btnPrintReportBottom) {
        btnPrintReportBottom.addEventListener("click", handlePrintReportClick);
    }

    // Download PDF Report (Top & Bottom)
    const btnDownloadPdf = document.getElementById("btn-download-pdf");
    const btnDownloadPdfBottom = document.getElementById("btn-download-pdf-bottom");

    async function handleDownloadPdfClick(e) {
        e.preventDefault();
        e.stopPropagation();
        if (!currentAnalysisRecordId) {
            showAlert("Please run an analysis first before downloading the PDF report.", "warning");
            return;
        }
        await downloadReportPdfAsync(currentAnalysisRecordId, currentAnalysisRefId, e.currentTarget);
    }

    if (btnDownloadPdf) {
        btnDownloadPdf.addEventListener("click", handleDownloadPdfClick);
    }
    if (btnDownloadPdfBottom) {
        btnDownloadPdfBottom.addEventListener("click", handleDownloadPdfClick);
    }

    // =========================================================================
    // Live Camera Manager
    // =========================================================================

    function showCameraError(message) {
        if (cameraAlert && cameraAlertText) {
            cameraAlertText.textContent = message;
            cameraAlert.style.display = "block";
            if (btnCameraFallback) btnCameraFallback.style.display = "inline-flex";
        }
    }

    function hideCameraError() {
        if (cameraAlert) {
            cameraAlert.style.display = "none";
        }
        if (btnCameraFallback) {
            btnCameraFallback.style.display = "none";
        }
    }

    function handleCameraError(err) {
        console.error("Camera access error:", err);
        let msg = "Could not activate camera. Please check your camera permissions.";
        if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
            msg = "Camera permission was denied. Please allow camera access in your browser address bar (lock/site settings icon) to capture photos directly.";
        } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
            msg = "No camera hardware detected on this device. You can choose an image file instead.";
        } else if (err.name === "NotReadableError" || err.name === "TrackStartError") {
            msg = "Camera hardware is currently in use by another application. Please close other camera programs and try again.";
        } else if (err.name === "OverconstrainedError" || err.name === "ConstraintNotSatisfiedError") {
            msg = "The requested camera settings could not be satisfied by your hardware.";
        }
        showCameraError(msg);
    }

    async function enumerateVideoDevices() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            videoDevices = devices.filter(d => d.kind === "videoinput");
            if (videoDevices.length > 1 && btnSwitchCamera) {
                btnSwitchCamera.style.display = "flex";
            } else if (btnSwitchCamera) {
                btnSwitchCamera.style.display = "none";
            }
        } catch (err) {
            console.warn("Could not enumerate camera devices:", err);
        }
    }

    async function startCameraStream(preferredDeviceId = null) {
        stopCameraStream();
        hideCameraError();

        const constraints = {
            audio: false,
            video: preferredDeviceId 
                ? { deviceId: { exact: preferredDeviceId }, width: { ideal: 1920 }, height: { ideal: 1080 } }
                : { facingMode: { ideal: currentFacingMode }, width: { ideal: 1920 }, height: { ideal: 1080 } }
        };

        try {
            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            cameraStream = stream;
            if (cameraVideo) {
                cameraVideo.srcObject = stream;
                await cameraVideo.play().catch(() => {});
            }
            await enumerateVideoDevices();
        } catch (primaryErr) {
            console.warn("Primary camera stream attempt failed, trying fallback constraints...", primaryErr);
            try {
                const fallbackStream = await navigator.mediaDevices.getUserMedia({ audio: false, video: true });
                cameraStream = fallbackStream;
                if (cameraVideo) {
                    cameraVideo.srcObject = fallbackStream;
                    await cameraVideo.play().catch(() => {});
                }
                await enumerateVideoDevices();
            } catch (err) {
                handleCameraError(err);
            }
        }
    }

    async function switchCamera() {
        if (videoDevices.length > 1) {
            currentDeviceIndex = (currentDeviceIndex + 1) % videoDevices.length;
            const targetDevice = videoDevices[currentDeviceIndex];
            await startCameraStream(targetDevice.deviceId);
        } else {
            currentFacingMode = currentFacingMode === "environment" ? "user" : "environment";
            await startCameraStream();
        }
    }

    function captureSnapshot() {
        if (!cameraVideo || !cameraVideo.videoWidth || !cameraVideo.videoHeight) {
            return;
        }

        // Trigger brief visual flash
        if (cameraViewport) {
            const flash = document.createElement("div");
            flash.className = "camera-flash-overlay";
            cameraViewport.appendChild(flash);
            setTimeout(() => {
                if (flash.parentNode) flash.parentNode.removeChild(flash);
            }, 250);
        }

        cameraCanvas.width = cameraVideo.videoWidth;
        cameraCanvas.height = cameraVideo.videoHeight;
        const ctx = cameraCanvas.getContext("2d");
        ctx.drawImage(cameraVideo, 0, 0, cameraCanvas.width, cameraCanvas.height);

        cameraCanvas.toBlob((blob) => {
            if (!blob) return;
            capturedBlob = blob;

            if (cameraPreviewImg.src && cameraPreviewImg.src.startsWith("blob:")) {
                URL.revokeObjectURL(cameraPreviewImg.src);
            }
            cameraPreviewImg.src = URL.createObjectURL(blob);
            cameraPreviewImg.style.display = "block";
            cameraVideo.style.display = "none";
            if (cameraReticle) cameraReticle.style.display = "none";
            if (btnSwitchCamera) btnSwitchCamera.style.display = "none";

            if (cameraLiveControls) cameraLiveControls.style.display = "none";
            if (cameraReviewControls) cameraReviewControls.style.display = "flex";
        }, "image/jpeg", 0.95);
    }

    function retakeSnapshot() {
        if (cameraPreviewImg && cameraPreviewImg.src && cameraPreviewImg.src.startsWith("blob:")) {
            URL.revokeObjectURL(cameraPreviewImg.src);
        }
        capturedBlob = null;
        if (cameraPreviewImg) cameraPreviewImg.style.display = "none";
        if (cameraVideo) cameraVideo.style.display = "block";
        if (cameraReticle) cameraReticle.style.display = "block";
        if (videoDevices.length > 1 && btnSwitchCamera) {
            btnSwitchCamera.style.display = "flex";
        }
        if (cameraReviewControls) cameraReviewControls.style.display = "none";
        if (cameraLiveControls) cameraLiveControls.style.display = "flex";
    }

    function acceptCapturedPhoto() {
        if (!capturedBlob) return;
        const now = new Date();
        const dateStr = now.toISOString().replace(/[-:T]/g, "").slice(0, 14);
        const file = new File([capturedBlob], `camera_${dateStr}.jpg`, {
            type: "image/jpeg",
            lastModified: Date.now()
        });

        lastInputSource = "camera";
        closeCameraModal();
        handleFileSelection(file);
    }

    function stopCameraStream() {
        if (cameraStream) {
            try {
                cameraStream.getTracks().forEach(track => {
                    track.stop();
                });
            } catch (_) {}
            cameraStream = null;
        }
        if (cameraVideo) {
            cameraVideo.srcObject = null;
        }
    }

    function openCameraModal() {
        if (!cameraModal) {
            if (cameraInput) cameraInput.click();
            return;
        }

        // Reset views
        hideCameraError();
        if (cameraPreviewImg && cameraPreviewImg.src && cameraPreviewImg.src.startsWith("blob:")) {
            URL.revokeObjectURL(cameraPreviewImg.src);
        }
        capturedBlob = null;
        if (cameraPreviewImg) cameraPreviewImg.style.display = "none";
        if (cameraVideo) cameraVideo.style.display = "block";
        if (cameraReticle) cameraReticle.style.display = "block";
        if (cameraReviewControls) cameraReviewControls.style.display = "none";
        if (cameraLiveControls) cameraLiveControls.style.display = "flex";

        cameraModal.style.display = "flex";
        cameraModal.setAttribute("aria-hidden", "false");

        // Validate secure context and mediaDevices support
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            showCameraError(
                "Live webcam streaming is unavailable in this browser environment. " +
                (!window.isSecureContext ? "Web browsers require HTTPS or localhost/127.0.0.1 for live camera access. " : "") +
                "You can still select an image or use your device's native camera."
            );
            return;
        }

        startCameraStream();
    }

    function closeCameraModal() {
        stopCameraStream();
        if (cameraPreviewImg && cameraPreviewImg.src && cameraPreviewImg.src.startsWith("blob:")) {
            URL.revokeObjectURL(cameraPreviewImg.src);
        }
        capturedBlob = null;
        if (cameraModal) {
            cameraModal.style.display = "none";
            cameraModal.setAttribute("aria-hidden", "true");
        }
        if (cameraPreviewImg) cameraPreviewImg.style.display = "none";
        if (cameraVideo) cameraVideo.style.display = "block";
        if (cameraReticle) cameraReticle.style.display = "block";
        if (cameraReviewControls) cameraReviewControls.style.display = "none";
        if (cameraLiveControls) cameraLiveControls.style.display = "flex";
        hideCameraError();
    }

    // Modal Control Bindings
    if (btnCameraCapture) btnCameraCapture.addEventListener("click", captureSnapshot);
    if (btnCameraRetake) btnCameraRetake.addEventListener("click", retakeSnapshot);
    if (btnCameraUse) btnCameraUse.addEventListener("click", acceptCapturedPhoto);
    if (btnCameraCancel) btnCameraCancel.addEventListener("click", closeCameraModal);
    if (btnCloseCamera) btnCloseCamera.addEventListener("click", closeCameraModal);
    if (btnSwitchCamera) btnSwitchCamera.addEventListener("click", switchCamera);

    if (btnCameraFallback) {
        btnCameraFallback.addEventListener("click", () => {
            closeCameraModal();
            if (cameraInput) {
                cameraInput.click();
            } else if (fileInput) {
                fileInput.click();
            }
        });
    }

    if (cameraModal) {
        cameraModal.addEventListener("click", (e) => {
            if (e.target === cameraModal) {
                closeCameraModal();
            }
        });
    }

    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && cameraModal && cameraModal.style.display === "flex") {
            closeCameraModal();
        }
    });

    window.addEventListener("beforeunload", stopCameraStream);
    window.addEventListener("pagehide", stopCameraStream);
    document.addEventListener("visibilitychange", () => {
        if (document.hidden && cameraStream) {
            stopCameraStream();
        }
    });

    // Execute Inference Pipeline
    if (btnAnalyze) {
        btnAnalyze.addEventListener("click", async () => {
            if (!selectedFile) {
                showAlert("Please choose or capture a skin lesion photograph first.", "warning");
                return;
            }

            clearAlert();
            btnAnalyze.disabled = true;
            if (loadingSpinner) loadingSpinner.style.display = "block";
            if (reportSection) reportSection.style.display = "none";

            const formData = new FormData();
            formData.append("file", selectedFile);

            try {
                const response = await fetch("/api/predict", {
                    method: "POST",
                    body: formData
                });

                if (!response.ok) {
                    let errMsg = `Server returned HTTP ${response.status}`;
                    try {
                        const errJson = await response.json();
                        if (errJson.detail) errMsg = errJson.detail;
                    } catch (_) {}
                    throw new Error(errMsg);
                }

                const data = await response.json();
                renderOnPageReport(data);

            } catch (err) {
                showAlert(`Inference failed: ${err.message}`, "danger");
                btnAnalyze.disabled = false;
            } finally {
                if (loadingSpinner) loadingSpinner.style.display = "none";
            }
        });
    }

    function handleFileSelection(file) {
        clearAlert();

        const validTypes = ["image/jpeg", "image/png", "image/jpg"];
        if (!validTypes.includes(file.type.toLowerCase()) && !file.name.match(/\.(jpe?g|png)$/i)) {
            showAlert("Invalid file format. Please upload a genuine JPEG or PNG dermoscopic photograph.", "danger");
            return;
        }

        if (file.size > 15 * 1024 * 1024) {
            showAlert("File is too large. Maximum supported image size is 15 MB.", "danger");
            return;
        }

        selectedFile = file;

        // Render preview
        const reader = new FileReader();
        reader.onload = (e) => {
            if (previewImg) previewImg.src = e.target.result;
            if (previewFilename) previewFilename.textContent = file.name;
            if (previewFilesize) previewFilesize.textContent = formatBytes(file.size);
            if (previewBox) previewBox.style.display = "flex";
            if (dropzone) dropzone.style.display = "none";
            if (btnAnalyze) btnAnalyze.disabled = false;

            // Trigger lightweight image quality check
            runQualityCheck(file);
        };
        reader.readAsDataURL(file);
    }

    async function runQualityCheck(file) {
        const qualityBox = document.getElementById("quality-box");
        const qualityText = document.getElementById("quality-text");
        const qualityIcon = document.getElementById("quality-icon");

        if (!qualityBox) return;

        qualityBox.style.display = "flex";
        qualityBox.className = "quality-feedback-box acceptable";
        if (qualityIcon) qualityIcon.textContent = "⏳";
        if (qualityText) qualityText.textContent = "Checking image sharpness and lighting...";

        try {
            const formData = new FormData();
            formData.append("file", file);

            const res = await fetch("/api/check-quality", {
                method: "POST",
                body: formData
            });

            if (!res.ok) throw new Error("Quality service unavailable");
            const qData = await res.json();

            qualityBox.className = `quality-feedback-box ${qData.status}`;

            if (qData.status === "acceptable") {
                if (qualityIcon) qualityIcon.textContent = "✔";
                if (qualityText) qualityText.textContent = qData.message;
            } else if (qData.status === "warning") {
                if (qualityIcon) qualityIcon.textContent = "⚠️";
                const details = qData.warnings.join(" ") + (qData.suggestions.length > 0 ? ` (${qData.suggestions[0]})` : "");
                if (qualityText) qualityText.innerHTML = `<strong>Quality Warning:</strong> ${escapeHtml(details)} <span style="font-size:12px;display:block;margin-top:2px;">You may still proceed with analysis.</span>`;
            } else {
                if (qualityIcon) qualityIcon.textContent = "✕";
                if (qualityText) qualityText.textContent = qData.message;
                if (btnAnalyze) btnAnalyze.disabled = true;
            }
        } catch (_) {
            qualityBox.style.display = "none";
        }
    }

    function resetUploadState() {
        selectedFile = null;
        currentAnalysisRecordId = null;
        currentAnalysisRefId = null;
        if (fileInput) fileInput.value = "";
        if (cameraInput) cameraInput.value = "";
        if (previewImg) previewImg.src = "";
        if (previewBox) previewBox.style.display = "none";
        if (dropzone) dropzone.style.display = "block";
        if (btnAnalyze) btnAnalyze.disabled = true;
        const qualityBox = document.getElementById("quality-box");
        if (qualityBox) qualityBox.style.display = "none";
        destroyAnalysisCharts();
    }
}

/* ==========================================================================
   3. Render On-Page Analysis Report
   ========================================================================== */
function renderOnPageReport(data) {
    const reportSection = document.getElementById("report-section");
    if (!reportSection) return;

    currentAnalysisRecordId = data.record_id;
    currentAnalysisRefId = data.reference_id || `REC${data.record_id}`;

    const pred = data.prediction || {};
    const seg = data.segmentation || {};
    const sev = data.severity || {};
    const cost = data.cost_estimation || {};
    const imgs = data.image_paths || {};
    const edu = data.disease_info || {};

    // 1. Report Header Strip
    const reportMeta = document.getElementById("report-meta-text");
    if (reportMeta) {
        reportMeta.textContent = `Report Reference: ${data.reference_id || '#' + data.record_id} • Analyzed on: ${data.created_at || 'Just now'}`;
    }

    // 2. Summary Stat Cards
    const diseaseNameEl = document.getElementById("res-disease-name");
    const diseaseCodeEl = document.getElementById("res-disease-code");
    const confidenceEl = document.getElementById("res-confidence");
    const concernBadge = document.getElementById("res-concern-badge");
    const concernRationale = document.getElementById("res-concern-rationale");
    const lesionAreaEl = document.getElementById("res-lesion-area");

    const dName = pred.disease_name || pred.predicted_disease_name || "Unknown";
    const dCode = pred.disease_code || pred.predicted_class_code || "--";
    if (diseaseNameEl) diseaseNameEl.textContent = dName;
    if (diseaseCodeEl) diseaseCodeEl.textContent = dCode;

    const confScore = pred.confidence_percentage !== undefined ? pred.confidence_percentage : (pred.confidence_score ? `${(pred.confidence_score * 100).toFixed(1)}%` : "--%");
    if (confidenceEl) confidenceEl.textContent = typeof confScore === "number" ? `${confScore.toFixed(1)}%` : confScore;

    const concernLevel = sev.level || sev.severity_grade || "Moderate Concern";
    if (concernBadge) {
        concernBadge.textContent = concernLevel;
        concernBadge.style.backgroundColor = mapConcernColor(concernLevel);
    }
    if (concernRationale) {
        concernRationale.textContent = sev.summary || sev.reason || sev.rationale || "Algorithmic preliminary triage indicator.";
    }

    const areaPct = seg.relative_lesion_area_percentage !== undefined
        ? seg.relative_lesion_area_percentage
        : (seg.relative_lesion_area_pct !== undefined ? seg.relative_lesion_area_pct : 0);
    if (lesionAreaEl) lesionAreaEl.textContent = `${Number(areaPct).toFixed(2)}%`;

    // 3. Top-3 Differential Progress Bars
    const diffListEl = document.getElementById("res-differential-list");
    const top3 = pred.top_3_predictions || pred.top_3 || [];
    if (diffListEl && top3.length > 0) {
        diffListEl.innerHTML = top3.map((item, idx) => {
            const pctVal = typeof item.percentage === "string" ? parseFloat(item.percentage) : item.percentage;
            const name = item.display_name || item.disease_name || item.class_name || item.class_code;
            const code = item.class_code || "";
            return `
                <div class="diff-row">
                    <div class="diff-labels">
                        <span class="diff-name">#${idx + 1}: ${escapeHtml(name)} (${escapeHtml(code)})</span>
                        <span class="diff-percentage">${typeof item.percentage === 'string' ? item.percentage : item.percentage.toFixed(2) + '%'}</span>
                    </div>
                    <div class="diff-bar-track">
                        <div class="diff-bar-progress" style="width: ${Math.min(100, Math.max(2, pctVal || 0))}%; background-color: ${idx === 0 ? 'var(--primary)' : 'var(--accent)'};"></div>
                    </div>
                </div>
            `;
        }).join("");
    }

    // 4. Multi-Modal Visual Matrix (4 Images)
    const imgOrig = document.getElementById("img-card-orig");
    const imgMask = document.getElementById("img-card-mask");
    const imgOverlay = document.getElementById("img-card-overlay");
    const imgAttn = document.getElementById("img-card-attention");

    const srcOrig = imgs.original || imgs.original_image || imgs.input_image || "";
    const srcMask = imgs.mask || imgs.binary_mask || "";
    const srcOver = imgs.overlay || imgs.overlay_image || "";
    const srcAttn = imgs.attention || imgs.attention_map || (data.attention_map ? data.attention_map.attention_path : "");

    if (imgOrig) imgOrig.src = srcOrig;
    if (imgMask) imgMask.src = srcMask;
    if (imgOverlay) imgOverlay.src = srcOver;
    if (imgAttn) imgAttn.src = srcAttn;

    // 5. Educational Disease Information
    const eduTitle = document.getElementById("edu-title");
    const eduDesc = document.getElementById("edu-description");
    const eduChars = document.getElementById("edu-characteristics");
    const eduClinical = document.getElementById("edu-clinical-section");

    if (eduTitle) eduTitle.textContent = edu.name ? `About ${edu.name} (${edu.code})` : `About ${dName}`;
    if (eduDesc) eduDesc.textContent = edu.short_description || "General educational information regarding this dermatological category.";
    
    if (eduChars && edu.common_characteristics) {
        eduChars.innerHTML = edu.common_characteristics.map(c => `
            <span class="feature-tag">${escapeHtml(c)}</span>
        `).join("");
    }

    if (eduClinical) {
        const diagList = (edu.diagnostic_approaches || []).slice(0, 2).join(" • ");
        const treatList = (edu.treatment_approaches || []).slice(0, 2).join(" • ");
        eduClinical.innerHTML = `
            <div style="margin-top: 0.6rem;">
                <strong>Clinical Evaluation:</strong> <span class="secondary-text">${escapeHtml(diagList || "Dermoscopic examination & biopsy.")}</span>
            </div>
            <div style="margin-top: 0.35rem;">
                <strong>General Management:</strong> <span class="secondary-text">${escapeHtml(treatList || "Specialist monitoring or clinical excision.")}</span>
            </div>
            <div style="margin-top: 0.35rem;">
                <strong>When to Seek Care:</strong> <span class="secondary-text">${escapeHtml(edu.when_to_seek_care || "Consult a dermatologist if changing or symptomatic.")}</span>
            </div>
        `;
    }

    // 6. Treatment Cost Breakdown Table
    const costTbody = document.getElementById("cost-table-tbody");
    const breakdown = cost.breakdown || (cost.breakdown_by_category ? Object.values(cost.cost_breakdown_by_category || cost.breakdown_by_category) : []);

    if (costTbody && breakdown.length > 0) {
        const rowsHtml = breakdown.map(item => {
            const catName = item.category || item.category_name || "";
            const desc = item.description || "";
            const rangeStr = item.formatted_range || `INR ${(item.min_cost || 0).toLocaleString()} – ${(item.max_cost || 0).toLocaleString()}`;
            return `
                <tr>
                    <td><strong>${escapeHtml(catName)}</strong></td>
                    <td class="secondary-text">${escapeHtml(desc)}</td>
                    <td style="text-align: right; font-weight: 600; font-family: monospace;">${escapeHtml(rangeStr)}</td>
                </tr>
            `;
        }).join("");

        const totalRange = cost.formatted_total_range || cost.formatted_total_range_inr || "";
        const totalRowHtml = `
            <tr class="total-row">
                <td>TOTAL ESTIMATED RANGE</td>
                <td style="font-size: 13px; font-weight: normal; color: var(--text-secondary);">Comprehensive benchmark across all 5 tiers</td>
                <td style="text-align: right; font-family: monospace;">${escapeHtml(totalRange)}</td>
            </tr>
        `;
        costTbody.innerHTML = rowsHtml + totalRowHtml;
    }

    // Render Animated Data Visualizations (Chart.js)
    renderAnalysisCharts(pred, seg, cost);

    // Display report and scroll down
    reportSection.style.display = "block";
    reportSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

/* ==========================================================================
   Chart.js Visualization Manager
   ========================================================================== */
let chartPredictionsInstance = null;
let chartCoverageInstance = null;
let chartCostInstance = null;

function destroyAnalysisCharts() {
    if (chartPredictionsInstance) {
        chartPredictionsInstance.destroy();
        chartPredictionsInstance = null;
    }
    if (chartCoverageInstance) {
        chartCoverageInstance.destroy();
        chartCoverageInstance = null;
    }
    if (chartCostInstance) {
        chartCostInstance.destroy();
        chartCostInstance = null;
    }
}

function renderAnalysisCharts(pred, seg, cost) {
    if (typeof Chart === "undefined") {
        console.warn("Chart.js is not loaded; skipping canvas chart rendering.");
        return;
    }

    // Motion preference detection
    const prefersReducedMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const animDuration = prefersReducedMotion ? 0 : 1000;

    // Destroy any existing chart instances to avoid canvas reuse errors
    destroyAnalysisCharts();

    // 1. Top-3 Disease Predictions Horizontal Bar Chart
    const ctxPred = document.getElementById("chart-predictions");
    const top3 = pred.top_3_predictions || pred.top_3 || [];
    if (ctxPred && top3.length > 0) {
        const labels = top3.map(item => {
            const name = item.display_name || item.disease_name || item.class_name || item.class_code || "";
            const code = item.class_code ? ` (${item.class_code})` : "";
            return name + code;
        });
        const values = top3.map(item => {
            const p = typeof item.percentage === "string" ? parseFloat(item.percentage) : (item.percentage !== undefined ? item.percentage : (item.probability || 0) * 100);
            return Number(p.toFixed(2));
        });
        // Palette: Sky Blue (#00AFE0), Orange (#FF8500), Light Blue (#AEE2F5)
        const palette = ["#00AFE0", "#FF8500", "#AEE2F5"];

        chartPredictionsInstance = new Chart(ctxPred, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [{
                    label: "Model Score (%)",
                    data: values,
                    backgroundColor: palette.slice(0, labels.length),
                    borderColor: "#FFFFFF",
                    borderWidth: 1.5,
                    borderRadius: 6
                }]
            },
            options: {
                indexAxis: "y",
                responsive: true,
                maintainAspectRatio: false,
                animation: {
                    duration: animDuration,
                    easing: "easeOutQuart"
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: "#1F2937",
                        titleFont: { family: "Inter", size: 13, weight: "700" },
                        bodyFont: { family: "Inter", size: 12 },
                        padding: 10,
                        cornerRadius: 6,
                        callbacks: {
                            label: (ctx) => ` Model Score: ${ctx.raw}%`
                        }
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        max: Math.min(100, Math.max(10, Math.ceil(Math.max(...values, 10) / 10) * 10)),
                        ticks: {
                            callback: (v) => v + "%",
                            font: { family: "Inter", size: 11 },
                            color: "#4B5563"
                        },
                        grid: { color: "#E5E7EB" },
                        title: {
                            display: true,
                            text: "Model Probability Score (%)",
                            color: "#6B7280",
                            font: { family: "Inter", size: 11, weight: "600" }
                        }
                    },
                    y: {
                        ticks: {
                            font: { family: "Inter", size: 12, weight: "600" },
                            color: "#1F2937"
                        },
                        grid: { display: false }
                    }
                }
            }
        });
    }

    // 2. Lesion Image Coverage Doughnut Chart
    const ctxCov = document.getElementById("chart-lesion-coverage");
    if (ctxCov) {
        const rawArea = seg.relative_lesion_area_percentage !== undefined
            ? seg.relative_lesion_area_percentage
            : (seg.relative_lesion_area_pct !== undefined ? seg.relative_lesion_area_pct : 0);
        const lesionPct = Math.min(100, Math.max(0, Number(rawArea)));
        const skinPct = Math.max(0, Number((100 - lesionPct).toFixed(2)));

        // Update DOM labels
        const doughVal = document.getElementById("doughnut-pct-val");
        const covLesion = document.getElementById("coverage-lesion-pct");
        const covSkin = document.getElementById("coverage-skin-pct");
        if (doughVal) doughVal.textContent = `${lesionPct.toFixed(2)}%`;
        if (covLesion) covLesion.textContent = `${lesionPct.toFixed(2)}%`;
        if (covSkin) covSkin.textContent = `${skinPct.toFixed(2)}%`;

        chartCoverageInstance = new Chart(ctxCov, {
            type: "doughnut",
            data: {
                labels: ["Lesion Coverage", "Surrounding Skin / Background"],
                datasets: [{
                    data: [Number(lesionPct.toFixed(2)), skinPct],
                    backgroundColor: ["#FF8500", "#AEE2F5"], // Orange & Light Blue
                    borderColor: ["#FFFFFF", "#FFFFFF"],
                    borderWidth: 2,
                    hoverOffset: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: "68%",
                animation: {
                    duration: animDuration,
                    easing: "easeOutQuart"
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: "#1F2937",
                        titleFont: { family: "Inter", size: 13, weight: "700" },
                        bodyFont: { family: "Inter", size: 12 },
                        padding: 10,
                        cornerRadius: 6,
                        callbacks: {
                            label: (ctx) => ` ${ctx.label}: ${ctx.raw}%`
                        }
                    }
                }
            }
        });
    }

    // 3. Treatment Cost Comparison Grouped Bar Chart
    const ctxCost = document.getElementById("chart-cost-comparison");
    const breakdown = cost.breakdown || (cost.breakdown_by_category ? Object.values(cost.cost_breakdown_by_category || cost.breakdown_by_category) : []);
    if (ctxCost && breakdown.length > 0) {
        const labels = breakdown.map(item => item.category || item.category_name || "Tier");
        const minCosts = breakdown.map(item => item.min_cost !== undefined ? item.min_cost : (item.cost_range_inr ? item.cost_range_inr.min : 0));
        const maxCosts = breakdown.map(item => item.max_cost !== undefined ? item.max_cost : (item.cost_range_inr ? item.cost_range_inr.max : 0));

        // Update Total Strip Banner
        const totalMin = cost.total_min !== undefined ? cost.total_min : minCosts.reduce((a, b) => a + b, 0);
        const totalMax = cost.total_max !== undefined ? cost.total_max : maxCosts.reduce((a, b) => a + b, 0);
        const totalMinEl = document.getElementById("cost-total-min");
        const totalMaxEl = document.getElementById("cost-total-max");
        if (totalMinEl) totalMinEl.textContent = `₹${totalMin.toLocaleString("en-IN")}`;
        if (totalMaxEl) totalMaxEl.textContent = `₹${totalMax.toLocaleString("en-IN")}`;

        chartCostInstance = new Chart(ctxCost, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [
                    {
                        label: "Minimum Cost (INR)",
                        data: minCosts,
                        backgroundColor: "#00AFE0", // Sky Blue
                        borderColor: "#FFFFFF",
                        borderWidth: 1.5,
                        borderRadius: 4
                    },
                    {
                        label: "Maximum Cost (INR)",
                        data: maxCosts,
                        backgroundColor: "#FF8500", // Orange
                        borderColor: "#FFFFFF",
                        borderWidth: 1.5,
                        borderRadius: 4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: {
                    duration: animDuration,
                    easing: "easeOutQuart"
                },
                plugins: {
                    legend: {
                        display: true,
                        position: "top",
                        labels: {
                            boxWidth: 14,
                            color: "#1F2937",
                            font: { family: "Inter", size: 12, weight: "600" }
                        }
                    },
                    tooltip: {
                        backgroundColor: "#1F2937",
                        titleFont: { family: "Inter", size: 13, weight: "700" },
                        bodyFont: { family: "Inter", size: 12 },
                        padding: 10,
                        cornerRadius: 6,
                        callbacks: {
                            label: (ctx) => ` ${ctx.dataset.label}: ₹${ctx.raw.toLocaleString("en-IN")}`
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            font: { family: "Inter", size: 11, weight: "600" },
                            color: "#1F2937"
                        },
                        grid: { display: false }
                    },
                    y: {
                        beginAtZero: true,
                        ticks: {
                            callback: (v) => "₹" + v.toLocaleString("en-IN"),
                            font: { family: "Inter", size: 11 },
                            color: "#4B5563"
                        },
                        grid: { color: "#E5E7EB" },
                        title: {
                            display: true,
                            text: "Estimated Cost (INR)",
                            color: "#6B7280",
                            font: { family: "Inter", size: 11, weight: "600" }
                        }
                    }
                }
            }
        });
    }
}

/* ==========================================================================
   4. Disease Library Page Logic
   ========================================================================== */
async function initDiseaseLibrary() {
    const grid = document.getElementById("library-grid");
    const searchInput = document.getElementById("library-search");
    const countText = document.getElementById("library-count-text");
    const modeTabs = document.querySelectorAll(".mode-tab-btn");

    if (!grid) return;

    let activeMode = "all";

    function filterAndRender() {
        const query = searchInput ? searchInput.value.toLowerCase().trim() : "";
        const filtered = cachedLibraryData.filter(d => {
            const matchesMode = (activeMode === "all") || (d.mode === activeMode);
            const matchesQuery = !query || (
                d.name.toLowerCase().includes(query) ||
                d.code.toLowerCase().includes(query) ||
                (d.short_description && d.short_description.toLowerCase().includes(query)) ||
                (d.category_type && d.category_type.toLowerCase().includes(query))
            );
            return matchesMode && matchesQuery;
        });

        renderLibraryCards(filtered);

        if (countText) {
            countText.textContent = `Showing ${filtered.length} of ${cachedLibraryData.length} categories`;
        }
    }

    try {
        const response = await fetch("/api/disease-library");
        if (!response.ok) throw new Error("Failed to load disease library data.");
        const data = await response.json();
        cachedLibraryData = data.diseases || [];

        filterAndRender();

        if (searchInput) {
            searchInput.addEventListener("input", filterAndRender);
        }

        if (modeTabs) {
            modeTabs.forEach(tab => {
                tab.addEventListener("click", () => {
                    modeTabs.forEach(t => {
                        t.classList.remove("btn-primary", "active");
                        t.classList.add("btn-secondary");
                    });
                    tab.classList.remove("btn-secondary");
                    tab.classList.add("btn-primary", "active");
                    activeMode = tab.getAttribute("data-mode") || "all";
                    filterAndRender();
                });
            });
        }
    } catch (err) {
        grid.innerHTML = `
            <div class="card empty-state-box" style="grid-column: 1 / -1;">
                <p style="color: var(--severity-high);">Failed to load library: ${escapeHtml(err.message)}</p>
            </div>
        `;
    }

    setupDiseaseModal();
}

function renderLibraryCards(diseases) {
    const grid = document.getElementById("library-grid");
    if (!grid) return;

    if (diseases.length === 0) {
        grid.innerHTML = `
            <div class="card empty-state-box" style="grid-column: 1 / -1;">
                <div class="empty-state-icon">🔍</div>
                <h3 class="card-heading">No matching categories found</h3>
                <p class="secondary-text">Try searching with a different term such as Melanoma, Nevi, or Carcinoma.</p>
            </div>
        `;
        return;
    }

    grid.innerHTML = diseases.map(d => {
        const imgSrc = `/static/img/library/${d.representative_image}`;
        const isAiSupported = Boolean(d.ai_supported);
        const statusText = d.support_status || (isAiSupported ? "AI Analysis Active" : "Educational Reference Only");
        const statusClass = isAiSupported ? "badge-ai-active" : "badge-edu-only";
        const statusIcon = isAiSupported ? "✓ " : "📖 ";
        return `
            <div class="library-card">
                <div class="library-card-img-wrap">
                    <img src="${imgSrc}" alt="${escapeHtml(d.name)}" loading="lazy">
                    <span class="library-badge-status ${statusClass}">${statusIcon}${escapeHtml(statusText)}</span>
                    <span class="library-badge-type">${escapeHtml(d.category_type)}</span>
                </div>
                <div class="library-card-body">
                    <h3 class="library-card-title">${escapeHtml(d.name)}</h3>
                    <div class="library-card-code">Category Code: ${escapeHtml(d.code)}</div>
                    <p class="library-card-desc">${escapeHtml(d.short_description)}</p>
                    <button type="button" class="btn btn-secondary btn-learn-more" data-code="${escapeHtml(d.code)}">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"></path>
                            <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"></path>
                        </svg>
                        Learn More
                    </button>
                </div>
            </div>
        `;
    }).join("");

    // Attach event listeners for modal
    document.querySelectorAll(".btn-learn-more").forEach(btn => {
        btn.addEventListener("click", () => {
            const code = btn.getAttribute("data-code");
            const item = cachedLibraryData.find(d => d.code === code);
            if (item) openDiseaseModal(item);
        });
    });
}

function setupDiseaseModal() {
    const modal = document.getElementById("disease-modal");
    const closeBtn = document.getElementById("btn-close-disease-modal");

    if (closeBtn && modal) {
        closeBtn.addEventListener("click", () => modal.style.display = "none");
        modal.addEventListener("click", (e) => {
            if (e.target === modal) modal.style.display = "none";
        });
        document.addEventListener("keydown", (e) => {
            if (e.key === "Escape" && modal.style.display === "flex") {
                modal.style.display = "none";
            }
        });
    }
}

function openDiseaseModal(d) {
    const modal = document.getElementById("disease-modal");
    if (!modal) return;

    const isAiSupported = Boolean(d.ai_supported);
    const statusText = d.support_status || (isAiSupported ? "AI Analysis Active" : "Educational Reference Only");
    const statusClass = isAiSupported ? "badge-ai-active" : "badge-edu-only";
    const statusIcon = isAiSupported ? "✓ " : "📖 ";

    document.getElementById("modal-disease-title").textContent = `${d.name} (${d.code})`;
    document.getElementById("modal-disease-type").innerHTML = `
        ${escapeHtml(d.category_type)} • Reference Concern: ${escapeHtml(d.risk_level)}
        <div style="margin-top: 4px;">
            <span class="modal-status-badge ${statusClass}">${statusIcon}${escapeHtml(statusText)}</span>
        </div>
    `;
    document.getElementById("modal-disease-img").src = `/static/img/library/${d.representative_image}`;
    const captionEl = document.getElementById("modal-disease-img-caption");
    if (captionEl) {
        captionEl.textContent = (d.mode === "common_skin")
            ? "Representative clinical photograph (Google SCIN / SkinDisNet archive)"
            : "Representative dermoscopic sample (HAM10000 archive)";
    }
    document.getElementById("modal-disease-desc").textContent = d.short_description;

    const charsEl = document.getElementById("modal-disease-chars");
    if (charsEl) {
        charsEl.innerHTML = (d.common_characteristics || []).map(c => `<li>${escapeHtml(c)}</li>`).join("");
    }

    const diagEl = document.getElementById("modal-disease-diag");
    if (diagEl) {
        diagEl.innerHTML = (d.diagnostic_approaches || []).map(a => `<li>${escapeHtml(a)}</li>`).join("");
    }

    const treatEl = document.getElementById("modal-disease-treat");
    if (treatEl) {
        treatEl.innerHTML = (d.treatment_approaches || []).map(t => `<li>${escapeHtml(t)}</li>`).join("");
    }

    document.getElementById("modal-disease-care").textContent = d.when_to_seek_care || "Consult a doctor for evaluation.";
    document.getElementById("modal-disease-sources").textContent = (d.sources || []).join(" • ");

    modal.style.display = "flex";
}

/* ==========================================================================
   5. Reports Page Logic (Search, Filter, Table, Cards & Safe Deletion)
   ========================================================================== */
async function initReportsManagement() {
    const tbody = document.getElementById("reports-tbody");
    if (!tbody) return;

    const searchInput = document.getElementById("report-search");
    const categorySelect = document.getElementById("filter-category");
    const dateInput = document.getElementById("filter-date");
    const resetBtn = document.getElementById("btn-reset-filters");
    const refreshBtn = document.getElementById("btn-refresh-reports");

    await fetchAndRenderReports();

    // Event listeners
    if (searchInput) searchInput.addEventListener("input", applyReportFilters);
    if (categorySelect) categorySelect.addEventListener("change", applyReportFilters);
    if (dateInput) dateInput.addEventListener("change", applyReportFilters);

    if (resetBtn) {
        resetBtn.addEventListener("click", () => {
            if (searchInput) searchInput.value = "";
            if (categorySelect) categorySelect.value = "";
            if (dateInput) dateInput.value = "";
            applyReportFilters();
        });
    }

    if (refreshBtn) {
        refreshBtn.addEventListener("click", () => fetchAndRenderReports());
    }

    setupReportModal();
}

async function fetchAndRenderReports() {
    const tbody = document.getElementById("reports-tbody");
    const totalBadge = document.getElementById("reports-total-badge");

    try {
        const res = await fetch("/api/history");
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        cachedReports = data.records || [];

        if (totalBadge) totalBadge.textContent = `${cachedReports.length} Reports`;

        applyReportFilters();

    } catch (err) {
        if (tbody) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="7" class="empty-state-box" style="color: var(--severity-high);">
                        Failed to load reports: ${escapeHtml(err.message)}
                    </td>
                </tr>
            `;
        }
    }
}

function applyReportFilters() {
    const searchVal = (document.getElementById("report-search")?.value || "").toLowerCase().trim();
    const catVal = (document.getElementById("filter-category")?.value || "").toUpperCase().trim();
    const dateVal = (document.getElementById("filter-date")?.value || "").trim();

    let filtered = [...cachedReports];

    if (searchVal) {
        filtered = filtered.filter(r => 
            (r.reference_id && r.reference_id.toLowerCase().includes(searchVal)) ||
            (r.predicted_disease && r.predicted_disease.toLowerCase().includes(searchVal)) ||
            (r.predicted_class_code && r.predicted_class_code.toLowerCase().includes(searchVal)) ||
            String(r.id).includes(searchVal)
        );
    }

    if (catVal) {
        filtered = filtered.filter(r => 
            (r.predicted_class_code && r.predicted_class_code.toUpperCase() === catVal) ||
            (r.predicted_disease && r.predicted_disease.toUpperCase().includes(catVal))
        );
    }

    if (dateVal) {
        filtered = filtered.filter(r => 
            r.created_at && r.created_at.startsWith(dateVal)
        );
    }

    renderReports(filtered);
}

function renderReports(records) {
    const tbody = document.getElementById("reports-tbody");
    const mobileContainer = document.getElementById("mobile-reports-container");
    const emptyState = document.getElementById("reports-empty-state");

    if (records.length === 0) {
        if (tbody) tbody.innerHTML = "";
        if (mobileContainer) mobileContainer.innerHTML = "";
        if (emptyState) emptyState.style.display = "block";
        return;
    }

    if (emptyState) emptyState.style.display = "none";

    // 1. Desktop Table Rows
    if (tbody) {
        tbody.innerHTML = records.map(r => {
            const confPct = r.confidence_score ? (r.confidence_score * 100).toFixed(1) + "%" : "N/A";
            const costRange = r.estimated_cost_min
                ? `₹${r.estimated_cost_min.toLocaleString()} – ₹${r.estimated_cost_max ? r.estimated_cost_max.toLocaleString() : ''}`
                : "N/A";
            const concernLevel = r.severity_grade || "Moderate Concern";
            const concernColor = mapConcernColor(concernLevel);

            return `
                <tr>
                    <td>
                        <strong style="font-family: monospace; font-size: 13px; color: var(--primary);">
                            ${escapeHtml(r.reference_id || '#' + r.id)}
                        </strong>
                    </td>
                    <td class="secondary-text" style="white-space: nowrap;">
                        ${escapeHtml(r.created_at ? r.created_at.substring(0, 16) : 'N/A')}
                    </td>
                    <td>
                        <strong>${escapeHtml(r.predicted_disease || 'Pending')}</strong>
                        ${r.predicted_class_code ? `<span class="secondary-text">(${escapeHtml(r.predicted_class_code)})</span>` : ''}
                    </td>
                    <td style="font-weight: 600;">${confPct}</td>
                    <td>
                        <span class="concern-badge" style="background-color: ${concernColor}; font-size: 12px; padding: 0.2rem 0.55rem;">
                            ${escapeHtml(concernLevel)}
                        </span>
                    </td>
                    <td style="font-family: monospace; font-weight: 600;">${costRange}</td>
                    <td style="text-align: center; white-space: nowrap;">
                        <button type="button" class="btn btn-secondary btn-view-report" data-id="${r.id}" style="padding: 0.35rem 0.65rem; font-size: 13px; min-height: 34px; margin-right: 0.25rem;">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
                                <circle cx="12" cy="12" r="3"></circle>
                            </svg>
                            View
                        </button>
                        <button type="button" class="btn btn-primary btn-download-report" data-id="${r.id}" data-ref="${escapeHtml(r.reference_id || '#' + r.id)}" style="padding: 0.35rem 0.65rem; font-size: 13px; min-height: 34px; margin-right: 0.25rem;">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                                <polyline points="14 2 14 8 20 8"></polyline>
                                <line x1="12" y1="18" x2="12" y2="12"></line>
                                <polyline points="9 15 12 18 15 15"></polyline>
                            </svg>
                            PDF
                        </button>
                        <button type="button" class="btn btn-danger btn-delete-report" data-id="${r.id}" data-ref="${escapeHtml(r.reference_id || '#' + r.id)}" style="padding: 0.35rem 0.65rem; font-size: 13px; min-height: 34px;">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <polyline points="3 6 5 6 21 6"></polyline>
                                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                            </svg>
                            Delete
                        </button>
                    </td>
                </tr>
            `;
        }).join("");
    }

    // 2. Mobile Cards
    if (mobileContainer) {
        mobileContainer.innerHTML = records.map(r => {
            const confPct = r.confidence_score ? (r.confidence_score * 100).toFixed(1) + "%" : "N/A";
            const costRange = r.estimated_cost_min
                ? `₹${r.estimated_cost_min.toLocaleString()} – ₹${r.estimated_cost_max ? r.estimated_cost_max.toLocaleString() : ''}`
                : "N/A";
            const concernLevel = r.severity_grade || "Moderate Concern";
            const concernColor = mapConcernColor(concernLevel);

            return `
                <div class="report-card-mobile">
                    <div class="report-card-mobile-header">
                        <strong style="font-family: monospace; color: var(--primary);">
                            ${escapeHtml(r.reference_id || '#' + r.id)}
                        </strong>
                        <span class="secondary-text" style="font-size: 12px;">${escapeHtml(r.created_at ? r.created_at.substring(0, 10) : '')}</span>
                    </div>
                    <div class="report-mobile-row">
                        <span class="secondary-text">Predicted Category:</span>
                        <strong>${escapeHtml(r.predicted_disease || 'Pending')}</strong>
                    </div>
                    <div class="report-mobile-row">
                        <span class="secondary-text">Confidence:</span>
                        <span>${confPct}</span>
                    </div>
                    <div class="report-mobile-row">
                        <span class="secondary-text">Concern Level:</span>
                        <span class="concern-badge" style="background-color: ${concernColor}; font-size: 11.5px; padding: 0.15rem 0.45rem;">
                            ${escapeHtml(concernLevel)}
                        </span>
                    </div>
                    <div class="report-mobile-row">
                        <span class="secondary-text">Cost Range:</span>
                        <strong style="font-family: monospace;">${costRange}</strong>
                    </div>
                    <div style="display: flex; gap: 0.5rem; margin-top: 0.85rem; border-top: 1px solid var(--border-light); padding-top: 0.65rem;">
                        <button type="button" class="btn btn-secondary btn-view-report" data-id="${r.id}" style="flex: 1; padding: 0.45rem; font-size: 13px; min-height: 36px;">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
                                <circle cx="12" cy="12" r="3"></circle>
                            </svg>
                            View
                        </button>
                        <button type="button" class="btn btn-primary btn-download-report" data-id="${r.id}" data-ref="${escapeHtml(r.reference_id || '#' + r.id)}" style="flex: 1; padding: 0.45rem; font-size: 13px; text-align: center; min-height: 36px;">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                                <polyline points="14 2 14 8 20 8"></polyline>
                                <line x1="12" y1="18" x2="12" y2="12"></line>
                                <polyline points="9 15 12 18 15 15"></polyline>
                            </svg>
                            PDF
                        </button>
                        <button type="button" class="btn btn-danger btn-delete-report" data-id="${r.id}" data-ref="${escapeHtml(r.reference_id || '#' + r.id)}" style="padding: 0.45rem 0.65rem; font-size: 13px; min-height: 36px;">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <polyline points="3 6 5 6 21 6"></polyline>
                                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                            </svg>
                            Delete
                        </button>
                    </div>
                </div>
            `;
        }).join("");
    }

    // Attach View handlers
    document.querySelectorAll(".btn-view-report").forEach(btn => {
        btn.addEventListener("click", () => {
            const id = btn.getAttribute("data-id");
            openReportModal(id);
        });
    });

    // Attach Download handlers
    document.querySelectorAll(".btn-download-report").forEach(btn => {
        btn.addEventListener("click", () => {
            const id = btn.getAttribute("data-id");
            const ref = btn.getAttribute("data-ref") || ("REC" + id);
            downloadReportPdfAsync(id, ref, btn);
        });
    });

    // Attach Delete handlers
    document.querySelectorAll(".btn-delete-report").forEach(btn => {
        btn.addEventListener("click", async () => {
            const id = btn.getAttribute("data-id");
            const ref = btn.getAttribute("data-ref");
            const confirmed = window.confirm(`Are you sure you want to delete report ${ref}? This will permanently remove the record and its generated output files.`);
            if (confirmed) {
                await deleteReportRecord(id);
            }
        });
    });
}

async function deleteReportRecord(recordId) {
    try {
        const res = await fetch(`/api/records/${recordId}`, {
            method: "DELETE"
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        alert(data.message || `Report #${recordId} deleted successfully.`);
        await fetchAndRenderReports();

    } catch (err) {
        alert(`Failed to delete report: ${err.message}`);
    }
}

function setupReportModal() {
    const modal = document.getElementById("report-detail-modal");
    const closeBtn = document.getElementById("btn-close-report-modal");
    const printBtn = document.getElementById("modal-btn-print");
    const modalDownloadBtn = document.getElementById("modal-btn-download");

    if (closeBtn && modal) {
        closeBtn.addEventListener("click", () => modal.style.display = "none");
        modal.addEventListener("click", (e) => {
            if (e.target === modal) modal.style.display = "none";
        });
        document.addEventListener("keydown", (e) => {
            if (e.key === "Escape" && modal.style.display === "flex") {
                modal.style.display = "none";
            }
        });
    }

    if (modalDownloadBtn) {
        modalDownloadBtn.addEventListener("click", () => {
            if (currentModalReportId) {
                downloadReportPdfAsync(currentModalReportId, currentModalReportRef || `REC${currentModalReportId}`, modalDownloadBtn);
            }
        });
    }

    if (printBtn) {
        printBtn.addEventListener("click", () => {
            window.print();
        });
    }
}

async function openReportModal(recordId) {
    const modal = document.getElementById("report-detail-modal");
    if (!modal) return;

    try {
        const res = await fetch(`/api/results/${recordId}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const r = await res.json();

        currentModalReportId = r.id;
        currentModalReportRef = r.reference_id || `REC${r.id}`;

        document.getElementById("modal-report-meta").textContent = 
            `Reference: ${r.reference_id || '#' + r.id} • Date: ${r.created_at || 'N/A'}`;
        document.getElementById("modal-stat-disease").textContent = 
            `${r.predicted_disease || 'Unknown'} (${r.predicted_class_code || ''})`;
        document.getElementById("modal-stat-confidence").textContent = 
            r.confidence_score ? `${(r.confidence_score * 100).toFixed(1)}%` : '--%';

        const concernBadge = document.getElementById("modal-stat-concern");
        const concernLevel = r.severity_grade || "Moderate Concern";
        concernBadge.textContent = concernLevel;
        concernBadge.style.backgroundColor = mapConcernColor(concernLevel);

        document.getElementById("modal-stat-area").textContent = 
            r.lesion_area_percentage !== null && r.lesion_area_percentage !== undefined ? `${Number(r.lesion_area_percentage).toFixed(2)}%` : '--%';
        document.getElementById("modal-stat-cost").textContent = 
            r.estimated_cost_min ? `₹${r.estimated_cost_min.toLocaleString()} – ₹${r.estimated_cost_max ? r.estimated_cost_max.toLocaleString() : ''}` : '₹--';

        // 4 Images
        document.getElementById("modal-img-orig").src = r.image_path ? `/${r.image_path.replace(/^\//, '')}` : "";
        document.getElementById("modal-img-mask").src = r.mask_path ? `/${r.mask_path.replace(/^\//, '')}` : "";
        document.getElementById("modal-img-overlay").src = r.overlay_path ? `/${r.overlay_path.replace(/^\//, '')}` : "";
        document.getElementById("modal-img-attention").src = r.attention_map_path ? `/${r.attention_map_path.replace(/^\//, '')}` : "";

        // Cost table
        const costTbody = document.getElementById("modal-cost-tbody");
        const breakdown = r.cost_breakdown?.breakdown || (r.cost_breakdown?.breakdown_by_category ? Object.values(r.cost_breakdown.breakdown_by_category) : []);

        if (costTbody && breakdown.length > 0) {
            costTbody.innerHTML = breakdown.map(item => `
                <tr>
                    <td><strong>${escapeHtml(item.category || item.category_name || '')}</strong></td>
                    <td class="secondary-text">${escapeHtml(item.description || '')}</td>
                    <td style="text-align: right; font-family: monospace;">${escapeHtml(item.formatted_range || '')}</td>
                </tr>
            `).join("");
        }

        modal.style.display = "flex";

    } catch (err) {
        alert(`Failed to load report details: ${err.message}`);
    }
}

/* ==========================================================================
   6. Lightbox Component for Click-to-Enlarge
   ========================================================================== */
function initLightbox() {
    const modal = document.getElementById("lightbox-modal");
    const closeBtn = document.getElementById("btn-close-lightbox");
    const lightboxImg = document.getElementById("lightbox-img");
    const lightboxTitle = document.getElementById("lightbox-title");

    if (!modal) return;

    if (closeBtn) {
        closeBtn.addEventListener("click", () => modal.style.display = "none");
    }

    modal.addEventListener("click", (e) => {
        if (e.target === modal) modal.style.display = "none";
    });

    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && modal.style.display === "flex") {
            modal.style.display = "none";
        }
    });

    // Attach to image cards dynamically
    document.querySelectorAll(".image-container-wrap").forEach(wrap => {
        wrap.addEventListener("click", () => {
            const img = wrap.querySelector("img");
            if (img && img.src && !img.src.endsWith("#")) {
                lightboxImg.src = img.src;
                if (lightboxTitle) lightboxTitle.textContent = wrap.getAttribute("data-label") || "Visual Artifact Preview";
                modal.style.display = "flex";
            }
        });
    });
}

/* ==========================================================================
   7. Helper Utilities
   ========================================================================== */
function showAlert(message, type = "info") {
    const alertBox = document.getElementById("alert-box");
    if (!alertBox) return;

    alertBox.style.display = "block";
    if (type === "danger") {
        alertBox.style.backgroundColor = "#FEF2F2";
        alertBox.style.color = "#991B1B";
        alertBox.style.border = "1px solid #FECACA";
    } else if (type === "warning") {
        alertBox.style.backgroundColor = "#FFFBEB";
        alertBox.style.color = "#92400E";
        alertBox.style.border = "1px solid #FDE68A";
    } else {
        alertBox.style.backgroundColor = "var(--bg-secondary)";
        alertBox.style.color = "var(--primary)";
        alertBox.style.border = "1px solid #BBF7D0";
    }
    alertBox.textContent = message;
}

function clearAlert() {
    const alertBox = document.getElementById("alert-box");
    if (alertBox) {
        alertBox.style.display = "none";
        alertBox.textContent = "";
    }
}

function mapConcernColor(level) {
    if (!level) return "var(--severity-und)";
    const l = level.toLowerCase();
    if (l.includes("low")) return "var(--severity-low)";
    if (l.includes("mod")) return "var(--severity-mod)";
    if (l.includes("high")) return "var(--severity-high)";
    return "var(--severity-und)";
}

function formatBytes(bytes, decimals = 1) {
    if (!bytes || bytes === 0) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

function escapeHtml(str) {
    if (!str) return "";
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}
