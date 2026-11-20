
const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const previewContainer = document.getElementById("previewContainer");
const fileName = document.getElementById("fileName");

const analyzeButton = document.getElementById("analyzeButton");
const removeButton = document.getElementById("removeButton");

const loading = document.getElementById("loading");
const result = document.getElementById("result");

const prediction = document.getElementById("prediction");
const confidence = document.getElementById("confidence");
const confidenceBar = document.getElementById("confidenceBar");

// Disease information elements
const symptoms = document.getElementById("symptoms");
const solutionList = document.getElementById("solution-list");
const prevention = document.getElementById("prevention");

const dropArea = document.getElementById("dropArea");


// ==========================
// IMAGE VALIDATION
// ==========================

function handleImage(file) {

    if (!file) return;

    const allowedTypes = [
        "image/jpeg",
        "image/png"
    ];

    if (!allowedTypes.includes(file.type)) {
        alert("Please upload a JPG, JPEG or PNG image.");
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        alert("Image size should be less than 10 MB.");
        return;
    }

    imageInput.files = createFileList(file);

    fileName.textContent = file.name;

    const reader = new FileReader();

    reader.onload = function(event) {

        preview.src = event.target.result;

        previewContainer.style.display = "block";

        analyzeButton.disabled = false;

    };

    reader.readAsDataURL(file);

    result.style.display = "none";

    confidenceBar.style.width = "0%";
}


// Helper for assigning dropped files
function createFileList(file) {

    const dataTransfer = new DataTransfer();

    dataTransfer.items.add(file);

    return dataTransfer.files;
}


// ==========================
// FILE INPUT
// ==========================

imageInput.addEventListener("change", function() {

    handleImage(this.files[0]);

});


// ==========================
// DRAG AND DROP
// ==========================

dropArea.addEventListener("dragover", function(event) {

    event.preventDefault();

    dropArea.classList.add("drag-active");

});

dropArea.addEventListener("dragleave", function() {

    dropArea.classList.remove("drag-active");

});

dropArea.addEventListener("drop", function(event) {

    event.preventDefault();

    dropArea.classList.remove("drag-active");

    const file = event.dataTransfer.files[0];

    handleImage(file);

});


// ==========================
// REMOVE IMAGE
// ==========================

removeButton.addEventListener("click", function() {

    imageInput.value = "";

    preview.src = "";

    previewContainer.style.display = "none";

    fileName.textContent = "No image selected";

    analyzeButton.disabled = true;

    result.style.display = "none";

    confidenceBar.style.width = "0%";

});


// ==========================
// ANALYZE IMAGE
// ==========================

analyzeButton.addEventListener("click", async function() {

    const file = imageInput.files[0];

    if (!file) {
        alert("Please select an image first.");
        return;
    }

    const formData = new FormData();

    formData.append("image", file);

    loading.style.display = "block";

    result.style.display = "none";

    analyzeButton.disabled = true;

    try {

        const response = await fetch("/predict", {

            method: "POST",

            body: formData

        });

        const data = await response.json();

        if (!response.ok || data.error) {

            throw new Error(data.error || "Unable to analyze image.");

        }

        if (
            !data.prediction ||
            !Number.isFinite(Number(data.confidence))
        ) {
            throw new Error("Invalid response received from server.");
        }

        const confidenceValue = Math.min(
            100,
            Math.max(0, Number(data.confidence))
        );

        // Display prediction
        prediction.textContent = data.prediction;

        // Display confidence
        confidence.textContent = confidenceValue.toFixed(2);

// ==========================
// DISPLAY DISEASE SOLUTIONS
// ==========================

// Display symptoms
symptoms.textContent = data.symptoms || "No symptoms information available.";

// Display recommended solutions
solutionList.innerHTML = "";

if (Array.isArray(data.solution)) {

    data.solution.forEach(function(item) {

        const li = document.createElement("li");

        li.textContent = item;

        solutionList.appendChild(li);

    });

} else {

    const li = document.createElement("li");

    li.textContent = "No specific solution available.";

    solutionList.appendChild(li);

}

// Display prevention tips
prevention.textContent = data.prevention || "No prevention information available.";


        // Show result
        result.style.display = "block";

        // Animate progress bar
        requestAnimationFrame(() => {

            confidenceBar.style.width = confidenceValue + "%";

        });

        // Scroll to result
        result.scrollIntoView({
            behavior: "smooth",
            block: "nearest"
        });

    }

    catch(error) {

        alert(error.message || "Something went wrong. Please try again.");

        console.error(error);

    }

    finally {

        loading.style.display = "none";

        analyzeButton.disabled = false;

    }

});
