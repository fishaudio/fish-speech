import fs from 'node:fs'
import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react-swc'
import tailwindcss from '@tailwindcss/vite'
import path from 'node:path'

function inlineEntryAssets(): Plugin {
  let resolvedOutDir = ''

  return {
    name: 'inline-entry-assets',
    apply: 'build',
    configResolved(config) {
      resolvedOutDir = path.resolve(config.root, config.build.outDir)
    },
    closeBundle() {
      const indexHtmlPath = path.join(resolvedOutDir, 'index.html')
      if (!fs.existsSync(indexHtmlPath)) {
        return
      }

      const filesToDelete = new Set<string>()
      const escapeInlineScript = (code: string) => code.replace(/<\/script/gi, '<\\/script')
      const escapeInlineStyle = (code: string) => code.replace(/<\/style/gi, '<\\/style')
      const normalizeFileName = (assetPath: string) =>
        assetPath.replace(/^\//, '').replace(/^\.\//, '')
      const readBuiltAsset = (assetPath: string) => {
        const fileName = normalizeFileName(assetPath)
        const absolutePath = path.join(resolvedOutDir, fileName)
        if (!fs.existsSync(absolutePath)) {
          return null
        }

        filesToDelete.add(absolutePath)
        return fs.readFileSync(absolutePath, 'utf8')
      }

      let html = fs.readFileSync(indexHtmlPath, 'utf8')

      html = html.replace(
        /<link rel="modulepreload"[^>]+href="([^"]+)"[^>]*>/g,
        (_fullMatch, href: string) => {
          const absolutePath = path.join(resolvedOutDir, normalizeFileName(href))
          if (fs.existsSync(absolutePath)) {
            filesToDelete.add(absolutePath)
          }
          return ''
        },
      )

      html = html.replace(
        /<link rel="stylesheet"[^>]+href="([^"]+)"[^>]*>/g,
        (fullMatch, href: string) => {
          const assetSource = readBuiltAsset(href)
          if (!assetSource) {
            return fullMatch
          }

          return `<style>${escapeInlineStyle(assetSource)}</style>`
        },
      )

      html = html.replace(
        /<script type="module"[^>]+src="([^"]+)"[^>]*><\/script>/g,
        (fullMatch, src: string) => {
          const chunkCode = readBuiltAsset(src)
          if (!chunkCode) {
            return fullMatch
          }

          return `<script type="module">${escapeInlineScript(chunkCode)}</script>`
        },
      )

      fs.writeFileSync(indexHtmlPath, html)

      for (const filePath of filesToDelete) {
        fs.rmSync(filePath, { force: true })
      }

      fs.rmSync(path.join(resolvedOutDir, 'vite.svg'), { force: true })
      fs.rmSync(path.join(resolvedOutDir, 'assets'), { recursive: true, force: true })
    },
  }
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss(), inlineEntryAssets()],
  publicDir: false,
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  build: {
    assetsInlineLimit: Number.MAX_SAFE_INTEGER,
    cssCodeSplit: false,
    modulePreload: false,
    rollupOptions: {
      output: {
        inlineDynamicImports: true,
      },
    },
  },
  server: {
    proxy: {
      '/v1': 'http://localhost:8888',
      '/v2': 'http://localhost:8888',
      '/health': 'http://localhost:8888',
    },
  },
})
