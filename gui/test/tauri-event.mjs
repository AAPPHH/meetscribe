export function listen(name, callback) {
  const listeners = globalThis.__tauri.listeners;
  (listeners[name] ??= []).push(callback);
  return Promise.resolve(() => {});
}
