const dropzone = document.querySelector(".dropzone");
const fileInput = document.querySelector("#image-input");

if (dropzone && fileInput) {
  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (event) => {
    const files = event.dataTransfer.files;
    if (files.length > 0) {
      fileInput.files = files;
      const subtitle = dropzone.querySelector(".dropzone-subtitle");
      if (subtitle) {
        subtitle.textContent = files[0].name;
      }
    }
  });

  fileInput.addEventListener("change", () => {
    const subtitle = dropzone.querySelector(".dropzone-subtitle");
    if (subtitle && fileInput.files.length > 0) {
      subtitle.textContent = fileInput.files[0].name;
    }
  });
}
