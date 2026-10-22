import { useState } from "react";
import { api } from "../api";

/** Shown instead of the app when the server asks for sign-in (it's reachable from other devices). */
export default function SignIn({ onDone }: { onDone: () => void }) {
  const [password, setPassword] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setErr(null);
    try {
      await api.post("/api/auth/login", { password });
      onDone();
    } catch (e: any) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="signin">
      <form className="card card-pad signin-card" onSubmit={submit}>
        <div className="brand">
          <img src="/logo.svg" alt="" />
          <div className="brand-name">Dainamo<span>Trainer</span></div>
        </div>
        <label className="field">
          <span>Access key or password</span>
          <input type="password" autoFocus autoComplete="current-password" value={password} onChange={(e) => setPassword(e.target.value)} />
        </label>
        {err && <div className="validation error">{err}</div>}
        <button className="btn primary" type="submit" disabled={busy || !password.trim()}>{busy ? "Signing in…" : "Sign in"}</button>
        <p className="help">
          This server can be reached from other devices, so it asks who you are. The access key is printed in
          the terminal where the server started, along with a link that signs you in.
        </p>
      </form>
    </div>
  );
}
