document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    const yearInput = document.getElementById('yearInput');
    const errorMessage = document.getElementById('errorMessage');
    const result = document.getElementById('result');
    const predictionValue = document.getElementById('predictionValue');
    const plotDiv = document.getElementById('plotDiv');

    // Set focus on the year input when page loads
    yearInput.focus();

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Reset UI elements
        errorMessage.style.display = 'none';
        result.style.display = 'none';
        
        // Validate year input
        const year = parseInt(yearInput.value);
        if (isNaN(year) || year < 2010 || year > 2050) {
            errorMessage.textContent = 'Please enter a valid year between 2010 and 2050';
            errorMessage.style.display = 'block';
            return;
        }

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({ year: year })
            });

            if (!response.ok) {
                const contentType = response.headers.get('content-type');
                if (contentType && contentType.includes('application/json')) {
                    const errorData = await response.json();
                    throw new Error(errorData.error || 'Prediction failed');
                } else {
                    throw new Error('Server error occurred');
                }
            }

            const data = await response.json();
            
            // Display prediction
            predictionValue.textContent = data.prediction.toLocaleString();
            result.style.display = 'block';

            // Update plot
            if (data.plot) {
                const plotData = JSON.parse(data.plot);
                Plotly.newPlot('plotDiv', plotData.data, plotData.layout);
            }
        } catch (error) {
            errorMessage.textContent = error.message;
            errorMessage.style.display = 'block';
        }
    });
});