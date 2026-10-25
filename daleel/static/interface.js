"use strict";

// Presentation helpers; all submissions use the existing CSRF-protected forms.
document.querySelectorAll("[data-password-toggle]").forEach((button) => {
  const input = document.getElementById(button.dataset.passwordToggle);
  if (!input || input.type !== "password") return;
  button.hidden = false;
  const slash = document.createElementNS("http://www.w3.org/2000/svg", "path");
  slash.setAttribute("d", "M1 1l22 22");
  slash.classList.add("eye-slash");
  button.querySelector("svg").append(slash);
  button.addEventListener("click", () => {
    const visible = input.type === "password";
    input.type = visible ? "text" : "password";
    button.classList.toggle("is-revealed", visible);
    button.setAttribute("aria-label", visible ? "Hide password" : "Show password");
    button.setAttribute("aria-pressed", String(visible));
  });
});

document.querySelectorAll("[data-file-feedback]").forEach((input) => {
  const output = document.getElementById(input.dataset.fileFeedback);
  const zone = input.closest(".file-panel");
  if (!output || !zone) return;
  const update = () => {
    const selected = input.files && input.files.length > 0;
    output.textContent = selected ? input.files[0].name : "";
    output.hidden = !selected;
    zone.classList.toggle("has-file", selected);
    zone.classList.toggle("is-filled", selected);
  };
  input.addEventListener("change", update);
  zone.addEventListener("dragover", event => {
    event.preventDefault();
    zone.classList.add("is-dragging");
  });
  zone.addEventListener("dragleave", () => zone.classList.remove("is-dragging"));
  zone.addEventListener("drop", event => {
    event.preventDefault();
    zone.classList.remove("is-dragging");
    if (event.dataTransfer.files.length !== 1) return;
    input.files = event.dataTransfer.files;
    update();
  });
  window.addEventListener("pageshow", update);
});

const upload = document.getElementById("mission-upload");
const submit = document.getElementById("submit-mission");
if (upload && submit) {
  const label = submit.querySelector("[data-submit-label]");
  const reset = () => {
    submit.disabled = false;
    label.textContent = "Submit mission";
    submit.removeAttribute("aria-busy");
  };
  window.addEventListener("pageshow", reset);
  upload.addEventListener("submit", () => {
    submit.disabled = true;
    label.textContent = "Validating and submitting…";
    submit.setAttribute("aria-busy", "true");
  });
}

const historySearch = document.querySelector("[data-history-search]");
if (historySearch) {
  const rows = document.querySelectorAll("[data-history-row]");
  const noMatches = document.querySelector("[data-history-empty]");
  historySearch.addEventListener("input", () => {
    const query = historySearch.value.trim().toLocaleLowerCase();
    let count = 0;
    rows.forEach(row => {
      row.hidden = !row.textContent.toLocaleLowerCase().includes(query);
      if (!row.hidden) count++;
    });
    if (noMatches) noMatches.hidden = count !== 0;
  });
  rows.forEach(row => row.addEventListener("click", event => {
    if (event.target.closest("a, button") || window.getSelection().toString()) return;
    window.location.assign(row.dataset.missionHref);
  }));
}

function bindEvidenceImages(root) {
  root.querySelectorAll(".observation-evidence").forEach(image => {
    const unavailable = () => {
      const frame = image.closest(".frame-image");
      if (!frame) return;
      image.closest(".observation-image-link").hidden = true;
      frame.querySelector(".evidence-notice").hidden = false;
    };
    image.addEventListener("error", unavailable);
    if (image.complete && image.naturalWidth === 0) unavailable();
  });
}
bindEvidenceImages(document);

const dialog = document.getElementById("evidence-dialog");
if (dialog && typeof dialog.showModal === "function") {
  const panel = document.getElementById("evidence-dialog-panel");
  let activeRow;
  const close = () => dialog.close();
  const showMapPoint = (number) => {
    window.daleelMap?.focus(number);
  };
  document.querySelectorAll("[data-open-observation]").forEach(row => {
    row.addEventListener("click", event => {
      event.preventDefault();
      activeRow = row;
      document.querySelectorAll(".detection").forEach(item => item.classList.toggle("is-selected", item === row));
      const card = row.closest(".observation-card");
      const details = card.querySelector(".observation-details");
      panel.replaceChildren(...Array.from(details.children, child => child.cloneNode(true)));
      panel.querySelectorAll("[id]").forEach(element => element.removeAttribute("id"));
      panel.querySelectorAll("[data-frame-close]").forEach(button => {
        button.hidden = false;
        button.addEventListener("click", close);
      });
      panel.querySelectorAll(".observation-evidence").forEach(image => image.loading = "eager");
      bindEvidenceImages(panel);
      panel.querySelectorAll("[data-map-observation]").forEach(link => link.addEventListener("click", () => {
        close();
        showMapPoint(link.dataset.mapObservation);
      }));
      document.body.classList.add("modal-open");
      dialog.setAttribute("aria-label", "Observation " + card.dataset.observationNumber + " evidence");
      dialog.showModal();
      window.daleelMap?.focus(card.dataset.observationNumber, false);
    });
  });
  dialog.addEventListener("click", event => { if (event.target === dialog) close(); });
  dialog.addEventListener("close", () => {
    document.body.classList.remove("modal-open");
    panel.replaceChildren();
    activeRow?.focus({preventScroll: true});
  });
  document.documentElement.classList.add("ui-enhanced");
  window.addEventListener("pageshow", () => { if (dialog.open) close(); });
}

const filterButtons = document.querySelectorAll("[data-agreement-filter]");
filterButtons.forEach(button => {
  button.addEventListener("click", () => {
    let count = 0;
    filterButtons.forEach(item => {
      item.setAttribute("aria-pressed", String(item === button));
      item.classList.toggle("is-active", item === button);
    });
    document.querySelectorAll(".observation-card").forEach(card => {
      const visible = button.dataset.agreementFilter === "all" || card.dataset.agreement === button.dataset.agreementFilter;
      card.hidden = !visible;
      if (visible) count++;
    });
    document.getElementById("visible-observation-count").textContent = count + " observations";
    window.daleelMap?.filter();
  });
});

// Store only a navigation preference, never account or mission data.
const sidebarToggle = document.querySelector("[data-sidebar-toggle]");
const sidebar = document.getElementById("primary-sidebar");
if (sidebarToggle && sidebar) {
  const storageKey = "daleel.sidebar.collapsed";
  const apply = collapsed => {
    document.body.classList.toggle("sidebar-collapsed", collapsed);
    sidebarToggle.setAttribute("aria-expanded", String(!collapsed));
    sidebarToggle.setAttribute("aria-label", collapsed ? "Expand sidebar" : "Collapse sidebar");
    sidebarToggle.title = collapsed ? "Expand sidebar" : "Collapse sidebar";
  };
  let collapsed = false;
  try { collapsed = localStorage.getItem(storageKey) === "true"; } catch (error) { /* Preference is optional. */ }
  apply(collapsed);
  sidebarToggle.hidden = false;
  sidebarToggle.addEventListener("click", () => {
    collapsed = !collapsed;
    apply(collapsed);
    try { localStorage.setItem(storageKey, String(collapsed)); } catch (error) { /* Navigation still works. */ }
  });
}


// Temporary credentials exist only in this one POST response, never storage.
const copyCredentials = document.querySelector("[data-copy-credentials]");
if (copyCredentials) {
  const username = document.getElementById("temporary-username");
  const password = document.getElementById("temporary-password");
  const status = document.querySelector("[data-copy-status]");
  copyCredentials.hidden = false;
  copyCredentials.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText("Username: " + username.value + "\nTemporary password: " + password.value);
      status.textContent = "Credentials copied. Share privately offline and clear your clipboard afterward.";
    } catch {
      password.focus();
      password.select();
      status.textContent = "Clipboard unavailable. Select and copy the details above.";
    }
  });
  window.addEventListener("pagehide", () => {
    username.value = "";
    password.value = "";
    copyCredentials.disabled = true;
    status.textContent = "Credentials cleared. Generate a new temporary password if needed.";
  });
}
