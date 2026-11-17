// svg_a_png.js — Convierte los SVG de arquitectura/ a PNG (para diapositivas).
// Uso: node herramientas/svg_a_png.js      (requiere Playwright con Chromium)
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

(async () => {
  const carpeta = path.join(__dirname, '..', 'arquitectura');
  const navegador = await chromium.launch();
  for (const archivo of fs.readdirSync(carpeta).filter((f) => f.endsWith('.svg'))) {
    const svg = fs.readFileSync(path.join(carpeta, archivo), 'utf8');
    const [, ancho, alto] = svg.match(/viewBox="0 0 (\d+) (\d+)"/).map(Number);
    const pagina = await navegador.newPage({ viewport: { width: ancho, height: alto }, deviceScaleFactor: 2 });
    await pagina.setContent(`<body style="margin:0">${svg}</body>`);
    await pagina.screenshot({ path: path.join(carpeta, archivo.replace('.svg', '.png')) });
    await pagina.close();
    console.log('PNG:', archivo.replace('.svg', '.png'));
  }
  await navegador.close();
})();
