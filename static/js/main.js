document.addEventListener('DOMContentLoaded', function () {

    const problemTextInput = document.getElementById('problem-text');
    const problemSelect = document.getElementById('problem-select');
    const modelSelectionContainer = document.getElementById('model-selection-container');
    const modelsLoading = document.getElementById('models-loading');
    const languageSelect = document.getElementById('language-select');
    const solveBtn = document.getElementById('solve-btn');

    const loadingSpinner = document.getElementById('loading-spinner');
    const resultsContainer = document.getElementById('results-container');

    // --- Problem Loading ---
    fetch('/api/problems')
        .then(response => response.json())
        .then(problems => {
            if (problems && problems.length > 0) {
                problems.forEach(problemFile => {
                    const option = document.createElement('option');
                    option.value = problemFile;
                    option.textContent = problemFile;
                    problemSelect.appendChild(option);
                });
            } else {
                problemSelect.disabled = true;
                problemSelect.querySelector('option').textContent = 'No problems found';
            }
        });
    
    problemSelect.addEventListener('change', () => {
        const selectedFile = problemSelect.value;
        if (selectedFile && selectedFile !== 'Select a problem file...') {
            fetch(`/api/problems/${selectedFile}`)
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Failed to load problem file.');
                    }
                    return response.json();
                })
                .then(data => {
                    problemTextInput.value = data.content;
                })
                .catch(error => {
                    console.error('Error loading problem:', error);
                    alert('Could not load the selected problem file.');
                });
        }
    });

    // --- Model Loading ---
    fetch('/api/models')
        .then(response => response.json())
        .then(models => {
            modelsLoading.classList.add('d-none');
            if (models && models.length > 0) {
                models.forEach(modelId => {
                    const div = document.createElement('div');
                    div.className = 'form-check form-check-inline';

                    const input = document.createElement('input');
                    input.className = 'form-check-input model-checkbox';
                    input.type = 'checkbox';
                    input.id = `check-${modelId}`;
                    input.value = modelId;

                    const label = document.createElement('label');
                    label.className = 'form-check-label';
                    label.htmlFor = `check-${modelId}`;
                    label.textContent = modelId;

                    div.appendChild(input);
                    div.appendChild(label);
                    modelSelectionContainer.appendChild(div);
                });
            } else {
                modelSelectionContainer.innerHTML = '<div class="form-text text-danger">Could not load models. Please configure the proxy in <a href="/settings">Settings</a>.</div>';
            }
        });


    // --- Problem Solving ---
    solveBtn.addEventListener('click', () => {
        const problem = problemTextInput.value;
        const selectedLanguage = languageSelect.value;
        const selectedModels = Array.from(document.querySelectorAll('.model-checkbox:checked'))
            .map(checkbox => checkbox.value);

        if (!problem.trim()) {
            alert('Please enter a problem statement.');
            return;
        }

        if (selectedModels.length === 0) {
            alert('Please select at least one model.');
            return;
        }

        resultsContainer.innerHTML = '';
        loadingSpinner.classList.remove('d-none');
        solveBtn.disabled = true;

        const payload = {
            problem: problem,
            models: selectedModels,
            language: selectedLanguage
        };

        // Create a streaming request using EventSource-compatible approach
        const eventSourceUrl = `/api/solve?${new URLSearchParams(payload)}`;
        
        // Since EventSource only supports GET requests, we need a different approach for POST with streaming
        fetch('/api/solve', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'text/event-stream'
            },
            body: JSON.stringify(payload),
            keepalive: true
        }).then(response => {
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            
            // Create result container for the single model (only first model is processed)
            const modelName = selectedModels[0];
                const resultCol = document.createElement('div');
            resultCol.className = 'col-12 mb-3';

                const resultCard = document.createElement('div');
                resultCard.className = 'card h-100';

                const cardHeader = document.createElement('div');
            cardHeader.className = 'card-header d-flex justify-content-between align-items-center';
            cardHeader.innerHTML = `<span>${modelName} - Agent Solving Process</span><div class="spinner-border spinner-border-sm text-primary" role="status"></div>`;

                const cardBody = document.createElement('div');
                cardBody.className = 'card-body';
            cardBody.innerHTML = '<div class="alert alert-info">Starting agent solving process...</div>';

                resultCard.appendChild(cardHeader);
                resultCard.appendChild(cardBody);
                resultCol.appendChild(resultCard);
                resultsContainer.appendChild(resultCol);

            let buffer = '';
            
            function processStream() {
                reader.read().then(({ done, value }) => {
                    if (done) {
                        loadingSpinner.classList.add('d-none');
                        solveBtn.disabled = false;
                        // Remove spinner from header
                        const spinner = cardHeader.querySelector('.spinner-border');
                        if (spinner) spinner.remove();
                        return;
                    }

                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n');
                    buffer = lines.pop(); // Keep incomplete line in buffer

                    lines.forEach(line => {
                        if (line.startsWith('data: ')) {
                            try {
                                const data = JSON.parse(line.slice(6)); // Remove 'data: ' prefix
                                handleStreamData(data, cardBody);
                            } catch (e) {
                                console.error('Failed to parse SSE data:', e, 'Line:', line);
                            }
                        }
                    });

                    processStream();
                }).catch(error => {
                    console.error('Stream reading error:', error);
                    loadingSpinner.classList.add('d-none');
                    solveBtn.disabled = false;
                    cardBody.innerHTML = `<div class="alert alert-danger">Stream error: ${error.message}</div>`;
                });
            }

            processStream();

        }).catch(error => {
            console.error('Error:', error);
            loadingSpinner.classList.add('d-none');
            solveBtn.disabled = false;
            resultsContainer.innerHTML = `<div class="alert alert-danger">An error occurred: ${error.message}</div>`;
        });

        function handleStreamData(data, cardBody) {
            switch (data.type) {
                case 'status':
                    // Update status
                    const existingStatus = cardBody.querySelector('.status-update');
                    if (existingStatus) {
                        existingStatus.textContent = data.content;
                    } else {
                        cardBody.innerHTML = `<div class="alert alert-info status-update">${data.content}</div>`;
                    }
                    break;

                case 'retry_status':
                    // Show retry status as a warning message
                    const existingRetry = cardBody.querySelector('.retry-status');
                    if (existingRetry) {
                        existingRetry.textContent = data.content;
                    } else {
                        const retryDiv = document.createElement('div');
                        retryDiv.className = 'alert alert-warning retry-status mt-2';
                        retryDiv.innerHTML = `<i class="fas fa-sync-alt fa-spin"></i> ${data.content}`;
                        
                        // Insert after status update if it exists
                        const statusUpdate = cardBody.querySelector('.status-update');
                        if (statusUpdate) {
                            statusUpdate.parentNode.insertBefore(retryDiv, statusUpdate.nextSibling);
                        } else {
                            cardBody.appendChild(retryDiv);
                        }
                    }
                    break;

                case 'intermediate_solution':
                case 'final_solution':
                    // Clear retry status when a solution is successfully received
                    const retryStatusElement = cardBody.querySelector('.retry-status');
                    if (retryStatusElement) {
                        retryStatusElement.remove();
                    }
                    
                    // Add new solution section
                    const solutionDiv = document.createElement('div');
                    solutionDiv.className = 'mt-3';
                    
                    const title = data.title || (data.type === 'final_solution' ? 'Final Solution' : 'Solution Update');
                    const alertClass = data.type === 'final_solution' ? 'alert-success' : 'alert-secondary';
                    
                    solutionDiv.innerHTML = `
                        <div class="alert ${alertClass}">
                            <h5>${title}</h5>
                            <div class="solution-content">${marked.parse(data.content)}</div>
                        </div>
                    `;
                    
                    cardBody.appendChild(solutionDiv);
                    
                    // Render math in the new content
                    const solutionContent = solutionDiv.querySelector('.solution-content');
                    renderMathInElement(solutionContent, {
                        delimiters: [
                            {left: '$', right: '$', display: false},
                            {left: '$$', right: '$$', display: true}
                        ]
                    });
                    
                    // Scroll to bottom
                    cardBody.scrollTop = cardBody.scrollHeight;
                    break;

                case 'bug_report':
                    // Add bug report section
                    const bugDiv = document.createElement('div');
                    bugDiv.className = 'mt-3';
                    bugDiv.innerHTML = `
                        <div class="alert alert-warning">
                            <h6>${data.title || 'Verification Issues Found'}</h6>
                            <div class="bug-content">${marked.parse(data.content)}</div>
                        </div>
                    `;
                    cardBody.appendChild(bugDiv);
                    
                    // Render math in bug report
                    const bugContent = bugDiv.querySelector('.bug-content');
                    renderMathInElement(bugContent, {
                        delimiters: [
                            {left: '$', right: '$', display: false},
                            {left: '$$', right: '$$', display: true}
                        ]
                    });
                    break;

                case 'error':
                    // Show error
                    cardBody.innerHTML = `<div class="alert alert-danger">Error: ${data.content}</div>`;
                    break;
            }
        }
    });
});
