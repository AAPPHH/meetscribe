import { build } from "esbuild";
import { cpSync, mkdirSync, rmSync } from "node:fs";

const testing = process.argv.includes("--test");
const out = testing ? "test/.build" : "dist";
const shared = { entryPoints: ["src/main.ts"], bundle: true, format: "esm", target: "es2022", outfile: `${out}/main.js` };
rmSync(out, { recursive: true, force: true });
mkdirSync(out, { recursive: true });
if (testing) {
  await build({ ...shared, alias: { "@tauri-apps/api/core": "./test/tauri-core.mjs", "@tauri-apps/api/event": "./test/tauri-event.mjs" } });
} else {
  await build({ ...shared, minify: true });
  for (const file of ["index.html", "style.css"]) cpSync(`src/${file}`, `dist/${file}`);
}
