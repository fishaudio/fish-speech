import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
    return twMerge(clsx(inputs))
}

export type Theme = 'light' | 'dark'

const THEME_STORAGE_KEY = 'fish-speech.theme.v1'

export function readTheme(): Theme {
    // Explicit URL override, handy for demos and shareable preview links.
    const forced = new URLSearchParams(window.location.search).get('theme')
    if (forced === 'dark' || forced === 'light') {
        return forced
    }
    try {
        const stored = window.localStorage.getItem(THEME_STORAGE_KEY)
        if (stored === 'dark' || stored === 'light') {
            return stored
        }
    } catch {
        // Storage disabled; fall back to the OS preference.
    }
    return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

export function applyTheme(theme: Theme): void {
    const root = document.documentElement
    root.classList.toggle('dark', theme === 'dark')
    root.style.colorScheme = theme
}

export function saveTheme(theme: Theme): void {
    applyTheme(theme)
    // A manual choice wins over any ?theme= override still in the address bar.
    try {
        const url = new URL(window.location.href)
        if (url.searchParams.has('theme')) {
            url.searchParams.delete('theme')
            window.history.replaceState(null, '', url)
        }
    } catch {
        // Ignore history failures; the in-memory choice still applies.
    }
    try {
        window.localStorage.setItem(THEME_STORAGE_KEY, theme)
    } catch {
        // Keep the in-memory choice when storage is unavailable.
    }
}
