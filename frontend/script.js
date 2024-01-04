function submitForm() {
    // Get the selected transaction type
    var transactionTypeElement = document.getElementById("transaction_type");
    var selectedTransactionType = transactionTypeElement.options[transactionTypeElement.selectedIndex].value;

    // Map transaction type to 1 or 0
    var transactionTypeValue = (selectedTransactionType === "Transfer" || selectedTransactionType === "Cash Out") ? 1 : 0;

    // Prepare data to be sent to the Flask server
    var formData = {
        "step": parseInt(document.getElementById("step").value),  // Parse the step value as an integer
        "type": transactionTypeValue,
        "amount": parseFloat(document.getElementById("amount").value),
        "oldbalanceOrg": parseFloat(document.getElementById("oldbalanceOrg").value),
        "selected_model": document.getElementById("selected_model").value

    };

    // Send data to the Flask server
    fetch('/api/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(formData)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Server error: ' + response.status);
        }
        return response.json();
    })
    .then(data => {
        // Handle the response data
        document.getElementById('predictionResult').textContent = 'Prediction: ' + data.prediction_text;
        document.getElementById('limeExplanation').innerHTML = 'LIME Explanation: ' + data.lime_explanation_html;

        // Assuming you have a container for the CFE Narrative
        document.getElementById('counterfactualExplanation').innerHTML =  data.narrative_html;

        // Render plots using Plotly
        if (data.radar_plot_html) {
            Plotly.newPlot('cfeRadialPlot', data.radar_plot_html);
        }
        if (data.bar_chart_html) {
            Plotly.newPlot('cfePlotly', data.bar_chart_html);
        }
    })
    .catch(error => {
        console.error('Fetch Error:', error);
        alert('Error: ' + error.message);
    });
}

