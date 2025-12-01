# Aftertrace – Project Overview

## One-line

Aftertrace is a web-first, mobile-first camera lab that turns your videos into trace/blob/optical-flow art and quietly shows you how trackable you are.

## Goals

1. Art-quality visuals (TouchDesigner / music-video level, not cheap filters).
2. Surveillance literacy (stats + visuals that reveal what the camera “knows” about you).
3. Low friction (opens from a link; works like an app via PWA).
4. My vibe: minimal, calm, slightly eerie, non-corporate.

## Audience

18–30, chronically online, into visual experiments, tech/AI, and slightly freaked out by surveillance but not doomscrolling about it.

## Stack

- Backend: Python 3.11+, FastAPI, Uvicorn, OpenCV, Librosa, (later MediaPipe).
- Frontend: Next.js (App Router), TypeScript, minimal CSS/Tailwind.

## Phases

1. Skeleton: monorepo, stub /process, simple upload page. ✅
2. Effect engine: real process_video (optical flow + drawing). ✅/in progress
3. Presets: config-based presets + minimal controls. ✅
4. Surveillance overlay & stats. ⬅ CURRENT
5. Education layer: copy + “how to be less trackable”.
6. PWA polish & performance.

## Design principles

- One primary action per screen.
- Big type, lots of space, one accent color.
- No dark patterns, no growth hacks.
- Privacy by default.
