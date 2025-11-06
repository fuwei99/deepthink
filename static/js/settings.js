document.addEventListener('DOMContentLoaded', function () {

    const baseUrlInput = document.getElementById('base-url');
    const apiKeyInput = document.getElementById('api-key');
    const saveBtn = document.getElementById('save-settings-btn');
    const saveStatus = document.getElementById('save-status');

    // Load existing settings on page load
    fetch('/api/settings')
        .then(response => response.json())
        .then(settings => {
            baseUrlInput.value = settings.base_url || '';
            apiKeyInput.value = settings.api_key || '';
        });

    // Save settings when button is clicked
    saveBtn.addEventListener('click', () => {
        const settings = {
            base_url: baseUrlInput.value,
            api_key: apiKeyInput.value
        };

        saveStatus.textContent = 'Saving...';
        fetch('/api/settings', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(settings)
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                saveStatus.textContent = 'Saved!';
            } else {
                saveStatus.textContent = 'Error saving settings.';
            }
            setTimeout(() => { saveStatus.textContent = ''; }, 3000);
        });
    });
});
