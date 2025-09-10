function copyToClipboard() {
    const codeBlock = document.getElementById("bibtex").innerText;
    navigator.clipboard.writeText(codeBlock).then(
        () => {
            const icon = document.getElementById("bibtex-copy-icon");
            icon.classList.remove("far", "fa-copy");
            icon.classList.add("fas", "fa-check", "has-text-success");
        },
        (err) => alert("Failed to copy: " + err)
    );
}