export function invoke(cmd, args) {
  const harness = globalThis.__tauri;
  harness.calls.push({ cmd, args });
  const handler = harness.handlers[cmd];
  return handler ? Promise.resolve().then(() => handler(args)) : Promise.reject(`Kein Handler für ${cmd}`);
}
