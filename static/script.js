document.getElementById('capture-btn').addEventListener('click', function () {
    fetch('/capture', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.path) {
            alert('Image captured and saved successfully!');
        } else {
            alert('Failed to capture image.');
        }
    })
    .catch(error => console.error('Error:', error));
});
