
$(document).ready(function() {
    // Handle form submission
    $("#prediction-form").submit(function(e) {
        e.preventDefault();
        
        // Show loading state
        $("button[type='submit']").prop("disabled", true).html('<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...');
        
        // Send prediction request
        $.ajax({
            url: "/predict",
            type: "POST",
            data: $(this).serialize(),
            success: function(response) {
                if (response.success) {
                    // Display results
                    $("#anxiety-result").text(response.anxiety_level);
                    $("#anxiety-confidence").text("Confidence: " + response.anxiety_confidence);
                    
                    $("#counseling-result").text(response.counseling_need);
                    $("#counseling-confidence").text("Confidence: " + response.counseling_confidence);
                    
                    // Set progress bars
                    let anxietyConfidence = parseFloat(response.anxiety_confidence);
                    $("#anxiety-progress").css("width", anxietyConfidence + "%");
                    
                    let counselingConfidence = parseFloat(response.counseling_confidence);
                    $("#counseling-progress").css("width", counselingConfidence + "%");
                    
                    // Show results
                    $("#results").show();
                    $("#error-message").hide();
                } else {
                    // Show error
                    $("#error-message").text(response.error).show();
                    $("#results").hide();
                }
            },
            error: function() {
                $("#error-message").text("Server error. Please try again later.").show();
                $("#results").hide();
            },
            complete: function() {
                // Reset button state
                $("button[type='submit']").prop("disabled", false).text("Predict");
            }
        });
    });
});
