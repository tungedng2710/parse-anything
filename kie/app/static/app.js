const state = {
  documentFile: null,
  templateFile: null,
  templateData: null,
  result: null,
  metadata: null,
  previewUrl: null,
};

let templateDraft = null;
let templateFilename = "template.json";
let activeEditorTab = "table";

const elements = {
  form: document.querySelector("#extract-form"),
  documentInput: document.querySelector("#document-input"),
  documentDropzone: document.querySelector("#document-dropzone"),
  documentSelection: document.querySelector("#document-selection"),
  documentName: document.querySelector("#document-name"),
  documentMeta: document.querySelector("#document-meta"),
  documentFileIcon: document.querySelector("#document-file-icon"),
  documentPreview: document.querySelector("#document-preview"),
  templateInput: document.querySelector("#template-input"),
  templateDropzone: document.querySelector("#template-dropzone"),
  templateSelection: document.querySelector("#template-selection"),
  templateName: document.querySelector("#template-name"),
  templateMeta: document.querySelector("#template-meta"),
  templatePreview: document.querySelector("#template-preview"),
  templateDialog: document.querySelector("#template-editor-dialog"),
  templateFields: document.querySelector("#template-fields"),
  templateTableEmpty: document.querySelector("#template-table-empty"),
  templateTablePanel: document.querySelector("#template-table-panel"),
  templateRawPanel: document.querySelector("#template-raw-panel"),
  templateRawInput: document.querySelector("#template-raw-input"),
  templateEditorError: document.querySelector("#template-editor-error"),
  instructions: document.querySelector("#instructions"),
  thinking: document.querySelector("#thinking"),
  extractButton: document.querySelector("#extract-button"),
  buttonLabel: document.querySelector(".button-label"),
  buttonLoading: document.querySelector(".button-loading"),
  formError: document.querySelector("#form-error"),
  resultEmpty: document.querySelector("#result-empty"),
  resultLoading: document.querySelector("#result-loading"),
  resultContent: document.querySelector("#result-content"),
  resultActions: document.querySelector("#result-actions"),
  resultSummary: document.querySelector("#result-summary"),
  dataTree: document.querySelector("#data-tree"),
  rawJson: document.querySelector("#raw-json"),
  visualPanel: document.querySelector("#visual-panel"),
  rawPanel: document.querySelector("#raw-panel"),
  statusDot: document.querySelector("#status-dot"),
  modelStatus: document.querySelector("#model-status"),
  toast: document.querySelector("#toast"),
};

const sampleTemplate = {
  "THÔNG TIN NGƯỜI BỆNH": {
    "Họ tên": "verbatim-string",
    "Ngày sinh": "verbatim-string",
    "Giới tính": "verbatim-string",
    "Mã bệnh nhân": "verbatim-string"
  },
  "THÔNG TIN KHÁM": {
    "Ngày khám": "verbatim-string",
    "Chẩn đoán": "verbatim-string",
    "Bác sĩ": "verbatim-string"
  },
  "KẾT QUẢ": [{
    "Tên chỉ số": "verbatim-string",
    "Kết quả": "verbatim-string",
    "Giá trị tham chiếu": "verbatim-string",
    "Đơn vị": "verbatim-string"
  }]
};

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("show");
  window.setTimeout(() => elements.toast.classList.remove("show"), 2200);
}

function showError(message) {
  elements.formError.textContent = message;
  elements.formError.classList.remove("hidden");
}

function clearError() {
  elements.formError.textContent = "";
  elements.formError.classList.add("hidden");
}

function setDocument(file) {
  clearError();
  const allowed = ["application/pdf", "image/png", "image/jpeg", "image/webp"];
  const extensionOkay = /\.(pdf|png|jpe?g|webp)$/i.test(file.name);
  if (!allowed.includes(file.type) && !extensionOkay) {
    showError("Choose a PDF, PNG, JPEG, or WebP document.");
    return;
  }
  if (file.size > 40 * 1024 * 1024) {
    showError("The document must be 40 MB or smaller.");
    return;
  }

  state.documentFile = file;
  elements.documentDropzone.classList.add("hidden");
  elements.documentSelection.classList.remove("hidden");
  elements.documentName.textContent = file.name;
  const isPdf = file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf");
  elements.documentMeta.textContent = `${isPdf ? "PDF document" : "Image"} · ${formatBytes(file.size)}`;
  elements.documentFileIcon.textContent = isPdf ? "PDF" : "IMG";
  elements.documentFileIcon.classList.toggle("image", !isPdf);

  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = URL.createObjectURL(file);
  elements.documentPreview.replaceChildren();
  const preview = isPdf ? document.createElement("object") : document.createElement("img");
  preview.data = isPdf ? state.previewUrl : undefined;
  preview.src = isPdf ? undefined : state.previewUrl;
  preview.type = isPdf ? "application/pdf" : undefined;
  preview.alt = isPdf ? "" : `Preview of ${file.name}`;
  elements.documentPreview.append(preview);
  elements.documentPreview.classList.remove("hidden");
}

function isTemplateObject(value, requireFields = true) {
  return Boolean(value) && !Array.isArray(value) && typeof value === "object" && (!requireFields || Object.keys(value).length > 0);
}

function cloneJson(value) {
  return JSON.parse(JSON.stringify(value));
}

function commitTemplate(parsed, filename = "template.json") {
  const serialized = JSON.stringify(parsed, null, 2);
  const safeFilename = filename.toLowerCase().endsWith(".json") ? filename : `${filename}.json`;
  const file = new File([serialized], safeFilename, { type: "application/json" });

  state.templateFile = file;
  state.templateData = cloneJson(parsed);
  elements.templateDropzone.classList.add("hidden");
  elements.templateSelection.classList.remove("hidden");
  elements.templatePreview.classList.remove("hidden");
  elements.templateName.textContent = safeFilename;
  elements.templateMeta.textContent = `${Object.keys(parsed).length} top-level fields · ${formatBytes(file.size)}`;
  elements.templatePreview.textContent = serialized;
}

async function setTemplate(file) {
  clearError();
  if (!file.name.toLowerCase().endsWith(".json") && file.type !== "application/json") {
    showError("Choose a JSON template file.");
    return;
  }
  if (file.size > 512 * 1024) {
    showError("The template must be 512 KB or smaller.");
    return;
  }

  let parsed;
  try {
    parsed = JSON.parse(await file.text());
    if (!isTemplateObject(parsed)) throw new Error();
  } catch {
    showError("The template is not a valid non-empty JSON object.");
    return;
  }

  commitTemplate(parsed, file.name);
  openTemplateEditor(parsed, file.name);
}

function showTemplateEditorError(message) {
  elements.templateEditorError.textContent = message;
  elements.templateEditorError.classList.remove("hidden");
}

function clearTemplateEditorError() {
  elements.templateEditorError.textContent = "";
  elements.templateEditorError.classList.add("hidden");
}

function syncTemplateRaw() {
  elements.templateRawInput.value = JSON.stringify(templateDraft, null, 2);
}

function valueAtPath(root, path) {
  return path.reduce((value, part) => value[part], root);
}

function parentAtPath(root, path) {
  return valueAtPath(root, path.slice(0, -1));
}

function flattenTemplate(value, path = [], entries = []) {
  const isObject = value !== null && typeof value === "object";
  const isPrimitiveArray = Array.isArray(value)
    && (value.length === 0 || value.every((item) => item === null || typeof item !== "object"));
  const isEmptyObject = isObject && !Array.isArray(value) && Object.keys(value).length === 0;

  if (!isObject || isPrimitiveArray || (isEmptyObject && path.length > 0)) {
    entries.push({ path, value });
    return entries;
  }
  if (isEmptyObject) return entries;

  Object.entries(value).forEach(([key, child]) => {
    const part = Array.isArray(value) ? Number(key) : key;
    flattenTemplate(child, [...path, part], entries);
  });
  return entries;
}

function pathLabel(path) {
  return path.map((part) => typeof part === "number" ? `Item ${part + 1}` : part).join(" › ");
}

function editorValue(value) {
  return typeof value === "string" ? value : JSON.stringify(value);
}

function deleteTemplatePath(path) {
  const parent = parentAtPath(templateDraft, path);
  const key = path[path.length - 1];
  if (Array.isArray(parent) && typeof key === "number") parent.splice(key, 1);
  else delete parent[key];
}

function renderTemplateRows() {
  elements.templateFields.replaceChildren();
  const entries = flattenTemplate(templateDraft);
  elements.templateTableEmpty.classList.toggle("hidden", entries.length > 0);

  entries.forEach((entry) => {
    const row = document.createElement("tr");
    const keyCell = document.createElement("td");
    const keyWrap = document.createElement("div");
    keyWrap.className = "template-key-cell";
    const parentPath = entry.path.slice(0, -1);
    if (parentPath.length) {
      const parentLabel = document.createElement("span");
      parentLabel.className = "template-parent-path";
      parentLabel.title = pathLabel(parentPath);
      parentLabel.textContent = pathLabel(parentPath);
      keyWrap.append(parentLabel);
    }

    const key = entry.path[entry.path.length - 1];
    const keyInput = document.createElement("input");
    keyInput.className = "template-cell-input template-key-input";
    keyInput.value = typeof key === "number" ? `Item ${key + 1}` : key;
    keyInput.disabled = typeof key === "number";
    keyInput.setAttribute("aria-label", `Field key ${pathLabel(entry.path)}`);
    keyInput.addEventListener("change", () => {
      const newKey = keyInput.value.trim();
      if (!newKey) {
        keyInput.value = key;
        return showTemplateEditorError("Field names cannot be empty.");
      }
      const parent = parentAtPath(templateDraft, entry.path);
      if (newKey !== key && Object.prototype.hasOwnProperty.call(parent, newKey)) {
        keyInput.value = key;
        return showTemplateEditorError(`A field named “${newKey}” already exists at this level.`);
      }
      if (newKey !== key) {
        parent[newKey] = parent[key];
        delete parent[key];
        clearTemplateEditorError();
        syncTemplateRaw();
        renderTemplateRows();
      }
    });
    keyWrap.append(keyInput);
    keyCell.append(keyWrap);

    const valueCell = document.createElement("td");
    const valueInput = document.createElement("input");
    valueInput.className = "template-cell-input";
    valueInput.value = editorValue(entry.value);
    valueInput.setAttribute("aria-label", `Type or value for ${pathLabel(entry.path)}`);
    if (typeof entry.value === "string") valueInput.setAttribute("list", "template-type-options");
    valueInput.addEventListener("change", () => {
      let nextValue = valueInput.value;
      if (typeof entry.value !== "string") {
        try {
          nextValue = JSON.parse(valueInput.value);
        } catch {
          valueInput.classList.add("invalid");
          return showTemplateEditorError("Arrays, objects, numbers, booleans, and null must remain valid JSON. Use Raw JSON to change their structure.");
        }
      }
      const parent = parentAtPath(templateDraft, entry.path);
      parent[entry.path[entry.path.length - 1]] = nextValue;
      valueInput.classList.remove("invalid");
      clearTemplateEditorError();
      syncTemplateRaw();
    });
    valueCell.append(valueInput);

    const actionCell = document.createElement("td");
    const deleteButton = document.createElement("button");
    deleteButton.className = "delete-field-button";
    deleteButton.type = "button";
    deleteButton.title = `Delete ${pathLabel(entry.path)}`;
    deleteButton.setAttribute("aria-label", `Delete ${pathLabel(entry.path)}`);
    deleteButton.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 7h14M9 7V4h6v3m-8 0 1 13h8l1-13M10 11v5M14 11v5" /></svg>';
    deleteButton.addEventListener("click", () => {
      deleteTemplatePath(entry.path);
      clearTemplateEditorError();
      syncTemplateRaw();
      renderTemplateRows();
    });
    actionCell.append(deleteButton);

    row.append(keyCell, valueCell, actionCell);
    elements.templateFields.append(row);
  });
}

function activateEditorTab(tabName) {
  activeEditorTab = tabName;
  document.querySelectorAll(".template-editor-tab").forEach((tab) => {
    const active = tab.dataset.editorTab === tabName;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
  });
  elements.templateTablePanel.classList.toggle("hidden", tabName !== "table");
  elements.templateRawPanel.classList.toggle("hidden", tabName !== "raw");
}

function readRawTemplate() {
  try {
    const parsed = JSON.parse(elements.templateRawInput.value);
    if (!isTemplateObject(parsed, false)) throw new Error("The root must be a JSON object.");
    templateDraft = parsed;
    clearTemplateEditorError();
    return true;
  } catch (error) {
    showTemplateEditorError(error.message || "Enter valid JSON.");
    return false;
  }
}

function selectEditorTab(tabName) {
  if (tabName === activeEditorTab) return;
  if (tabName === "table" && activeEditorTab === "raw") {
    if (!readRawTemplate()) return;
    renderTemplateRows();
  }
  if (tabName === "raw") syncTemplateRaw();
  activateEditorTab(tabName);
}

function openTemplateEditor(template, filename = "template.json") {
  templateDraft = cloneJson(template);
  templateFilename = filename;
  syncTemplateRaw();
  renderTemplateRows();
  clearTemplateEditorError();
  activateEditorTab("table");
  elements.templateDialog.classList.remove("hidden");
  document.body.classList.add("modal-open");
  document.querySelector("#close-template-editor").focus();
}

function closeTemplateEditor() {
  elements.templateDialog.classList.add("hidden");
  document.body.classList.remove("modal-open");
}

function applyTemplateEditor() {
  if (activeEditorTab === "raw" && !readRawTemplate()) return;
  if (!isTemplateObject(templateDraft)) {
    return showTemplateEditorError("The template must contain at least one top-level field.");
  }
  commitTemplate(templateDraft, templateFilename);
  closeTemplateEditor();
  clearError();
  showToast("Template updated");
}

function removeDocument() {
  state.documentFile = null;
  elements.documentInput.value = "";
  elements.documentDropzone.classList.remove("hidden");
  elements.documentSelection.classList.add("hidden");
  elements.documentPreview.classList.add("hidden");
  elements.documentPreview.replaceChildren();
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = null;
}

function removeTemplate() {
  state.templateFile = null;
  state.templateData = null;
  elements.templateInput.value = "";
  elements.templateDropzone.classList.remove("hidden");
  elements.templateSelection.classList.add("hidden");
  elements.templatePreview.classList.add("hidden");
  elements.templatePreview.textContent = "";
}

function bindDropzone(dropzone, input, setter) {
  dropzone.addEventListener("click", () => input.click());
  dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      input.click();
    }
  });
  input.addEventListener("change", () => input.files[0] && setter(input.files[0]));
  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.add("dragging");
    });
  });
  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.remove("dragging");
    });
  });
  dropzone.addEventListener("drop", (event) => {
    const file = event.dataTransfer.files[0];
    if (file) setter(file);
  });
}

function valueNode(value) {
  const span = document.createElement("span");
  span.className = "kv-value";
  if (value === null || value === undefined || value === "") {
    span.classList.add("null-value");
    span.textContent = "Not found";
  } else if (typeof value === "boolean") {
    span.classList.add("boolean-value");
    if (!value) span.classList.add("false");
    span.textContent = value ? "True" : "False";
  } else {
    span.textContent = String(value);
  }
  return span;
}

function renderKeyValueObject(object, container) {
  const simpleEntries = Object.entries(object).filter(([, value]) => value === null || typeof value !== "object");
  if (simpleEntries.length) {
    const list = document.createElement("div");
    list.className = "kv-list";
    simpleEntries.forEach(([key, value]) => {
      const row = document.createElement("div");
      row.className = "kv-row";
      const keyElement = document.createElement("span");
      keyElement.className = "kv-key";
      keyElement.textContent = key;
      row.append(keyElement, valueNode(value));
      list.append(row);
    });
    container.append(list);
  }

  Object.entries(object)
    .filter(([, value]) => value !== null && typeof value === "object")
    .forEach(([key, value]) => {
      const block = document.createElement("div");
      block.className = "nested-block";
      const title = document.createElement("p");
      title.className = "nested-title";
      title.textContent = key;
      block.append(title);
      renderComplexValue(value, block);
      container.append(block);
    });
}

function renderComplexValue(value, container) {
  if (Array.isArray(value)) {
    if (!value.length) {
      container.append(valueNode(null));
      return;
    }
    if (value.every((item) => item === null || typeof item !== "object")) {
      const list = document.createElement("div");
      list.className = "value-list";
      value.forEach((item) => {
        const tag = document.createElement("span");
        tag.className = "value-tag";
        tag.textContent = item === null ? "Not found" : String(item);
        list.append(tag);
      });
      container.append(list);
      return;
    }
    const stack = document.createElement("div");
    stack.className = "array-stack";
    value.forEach((item, index) => {
      const card = document.createElement("div");
      card.className = "array-item";
      const label = document.createElement("div");
      label.className = "array-index";
      label.textContent = `Item ${index + 1}`;
      card.append(label);
      if (item && typeof item === "object") renderKeyValueObject(item, card);
      else card.append(valueNode(item));
      stack.append(card);
    });
    container.append(stack);
    return;
  }
  if (value && typeof value === "object") renderKeyValueObject(value, container);
  else container.append(valueNode(value));
}

function renderResult(result, metadata) {
  state.result = result;
  state.metadata = metadata;
  elements.dataTree.replaceChildren();

  if (result && typeof result === "object" && !Array.isArray(result)) {
    Object.entries(result).forEach(([key, value]) => {
      const section = document.createElement("section");
      section.className = "data-section";
      const title = document.createElement("h3");
      title.className = "section-title";
      title.textContent = key;
      section.append(title);
      renderComplexValue(value, section);
      elements.dataTree.append(section);
    });
  } else {
    const section = document.createElement("section");
    section.className = "data-section";
    renderComplexValue(result, section);
    elements.dataTree.append(section);
  }

  elements.rawJson.textContent = JSON.stringify(result, null, 2);
  const seconds = (metadata.elapsed_ms / 1000).toFixed(1);
  elements.resultSummary.replaceChildren();
  [
    `${metadata.pages} ${metadata.pages === 1 ? "page" : "pages"}`,
    `${seconds}s`,
    metadata.completion_tokens ? `${metadata.completion_tokens} tokens` : null,
  ].filter(Boolean).forEach((label) => {
    const chip = document.createElement("span");
    chip.className = "summary-chip";
    chip.textContent = label;
    elements.resultSummary.append(chip);
  });

  elements.resultEmpty.classList.add("hidden");
  elements.resultLoading.classList.add("hidden");
  elements.resultContent.classList.remove("hidden");
  elements.resultActions.classList.remove("hidden");
}

function setLoading(loading) {
  elements.extractButton.disabled = loading;
  elements.buttonLabel.classList.toggle("hidden", loading);
  elements.buttonLoading.classList.toggle("hidden", !loading);
  if (loading) {
    elements.resultEmpty.classList.add("hidden");
    elements.resultContent.classList.add("hidden");
    elements.resultActions.classList.add("hidden");
    elements.resultLoading.classList.remove("hidden");
  }
}

async function extractDocument(event) {
  event.preventDefault();
  clearError();
  if (!state.documentFile) return showError("Add a source document before extracting.");
  if (!state.templateFile) return showError("Add a JSON extraction template.");

  const formData = new FormData();
  formData.append("document", state.documentFile);
  formData.append("template", state.templateFile);
  formData.append("instructions", elements.instructions.value.trim());
  formData.append("thinking", String(elements.thinking.checked));
  setLoading(true);

  try {
    const response = await fetch("/api/extract", { method: "POST", body: formData });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || `Extraction failed (HTTP ${response.status}).`);
    renderResult(payload.result, payload.metadata);
  } catch (error) {
    elements.resultLoading.classList.add("hidden");
    if (state.result) elements.resultContent.classList.remove("hidden");
    else elements.resultEmpty.classList.remove("hidden");
    showError(error.message || "Extraction failed. Please try again.");
  } finally {
    setLoading(false);
  }
}

function resetAll() {
  removeDocument();
  removeTemplate();
  state.result = null;
  state.metadata = null;
  elements.instructions.value = "";
  elements.thinking.checked = false;
  elements.resultContent.classList.add("hidden");
  elements.resultActions.classList.add("hidden");
  elements.resultLoading.classList.add("hidden");
  elements.resultEmpty.classList.remove("hidden");
  clearError();
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    const health = await response.json();
    const ready = health.model_server === "ready";
    elements.statusDot.classList.add(ready ? "ready" : "error");
    elements.modelStatus.textContent = ready ? "Model ready" : "Model offline";
  } catch {
    elements.statusDot.classList.add("error");
    elements.modelStatus.textContent = "App offline";
  }
}

bindDropzone(elements.documentDropzone, elements.documentInput, setDocument);
bindDropzone(elements.templateDropzone, elements.templateInput, setTemplate);
elements.form.addEventListener("submit", extractDocument);
document.querySelector("#remove-document").addEventListener("click", removeDocument);
document.querySelector("#remove-template").addEventListener("click", removeTemplate);
document.querySelector("#reset-button").addEventListener("click", resetAll);
document.querySelector("#type-template").addEventListener("click", () => {
  openTemplateEditor({ field_name: "verbatim-string" }, "template.json");
});
document.querySelector("#edit-template").addEventListener("click", () => {
  if (state.templateData) openTemplateEditor(state.templateData, state.templateFile?.name || "template.json");
});
document.querySelector("#sample-template").addEventListener("click", () => {
  commitTemplate(sampleTemplate, "medical-template.json");
  openTemplateEditor(sampleTemplate, "medical-template.json");
});
document.querySelector("#close-template-editor").addEventListener("click", closeTemplateEditor);
document.querySelector("#cancel-template-editor").addEventListener("click", closeTemplateEditor);
document.querySelector("#apply-template-editor").addEventListener("click", applyTemplateEditor);
document.querySelector("#add-template-field").addEventListener("click", () => {
  let index = 1;
  let key = "field_name";
  while (Object.prototype.hasOwnProperty.call(templateDraft, key)) key = `field_name_${++index}`;
  templateDraft[key] = "verbatim-string";
  clearTemplateEditorError();
  syncTemplateRaw();
  renderTemplateRows();
  const keyInputs = elements.templateFields.querySelectorAll(".template-key-input:not(:disabled)");
  keyInputs[keyInputs.length - 1]?.select();
});
document.querySelectorAll(".template-editor-tab").forEach((tab) => {
  tab.addEventListener("click", () => selectEditorTab(tab.dataset.editorTab));
});
elements.templateDialog.addEventListener("click", (event) => {
  if (event.target === elements.templateDialog) closeTemplateEditor();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !elements.templateDialog.classList.contains("hidden")) closeTemplateEditor();
});

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((item) => item.classList.toggle("active", item === tab));
    const showRaw = tab.dataset.tab === "raw";
    elements.visualPanel.classList.toggle("hidden", showRaw);
    elements.rawPanel.classList.toggle("hidden", !showRaw);
  });
});

document.querySelector("#copy-button").addEventListener("click", async () => {
  if (!state.result) return;
  await navigator.clipboard.writeText(JSON.stringify(state.result, null, 2));
  showToast("JSON copied to clipboard");
});

document.querySelector("#download-button").addEventListener("click", () => {
  if (!state.result) return;
  const blob = new Blob([JSON.stringify(state.result, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  const sourceName = (state.metadata?.filename || "document").replace(/\.[^.]+$/, "");
  anchor.href = url;
  anchor.download = `${sourceName}.extracted.json`;
  anchor.click();
  URL.revokeObjectURL(url);
  showToast("Result downloaded");
});

checkHealth();
