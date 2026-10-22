import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { ToastProvider } from "./components/ui";
import "./styles.css";

// a focused number field steps its value on the mouse wheel (Chrome, Edge), so scrolling the page past
// one quietly changes a setting; letting go of the field first makes the wheel scroll the page instead
document.addEventListener("wheel", (e) => {
  const el = document.activeElement;
  if (el instanceof HTMLInputElement && el.type === "number" && e.target === el) el.blur();
}, { passive: true });

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ToastProvider>
      <App />
    </ToastProvider>
  </StrictMode>,
);
