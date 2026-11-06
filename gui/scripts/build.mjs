import { build } from "esbuild";
import { cpSync, mkdirSync, rmSync } from "node:fs";

rmSync("dist", { recursive: true, force: true });
mkdirSync("dist", { recursive: true });
await build({ entryPoints: ["src/main.ts"], bundle: true, format: "esm", target: "es2022", outfile: "dist/main.js", minify: true });
for (const file of ["index.html", "style.css"]) cpSync(`src/${file}`, `dist/${file}`);
