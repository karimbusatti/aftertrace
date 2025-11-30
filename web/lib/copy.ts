/**
 * Copy configuration for Aftertrace stats and education layer.
 * 
 * Tone: calm, online 20–30 y/o, slightly poetic but clear.
 * No corporate buzzwords. Honest about what we do and don't know.
 */

// =============================================================================
// STATS COPY
// Labels and explainers for the surveillance stats
// =============================================================================

export const statsCopy = {
  trackability_score: {
    label: "trackability",
    explainer: "a rough 0–100 guess of how easy it is to follow you across frames. higher = more visible to algorithms.",
  },
  longest_track_seconds: {
    label: "longest track",
    explainer: "how long we kept eyes on a single point on you. longer = steadier signal for anyone watching.",
  },
  max_continuous_tracking_frames: {
    label: "max frames tracked",
    explainer: "the most frames a single point survived. more frames = more data to work with.",
  },
  average_points_per_frame: {
    label: "avg points per frame",
    explainer: "how many trackable features we found on you at any given moment. more points = richer profile.",
  },
  people_detected: {
    label: "subjects detected",
    explainer: "how many distinct people (roughly) appeared in frame. this is a stub—real detection coming later.",
  },
  frames_processed: {
    label: "frames",
    explainer: "total video frames we analyzed. more frames = more chances to learn your patterns.",
  },
  total_points_spawned: {
    label: "points tracked",
    explainer: "total number of trackable features we spawned and followed throughout the clip.",
  },
  beats_detected: {
    label: "beats",
    explainer: "audio peaks we used to sync the visual effect. no beats? we guessed timing instead.",
  },
} as const;

export type StatKey = keyof typeof statsCopy;

// =============================================================================
// TRACKING SUMMARY
// One-liner summaries based on trackability score ranges
// =============================================================================

export const trackingSummary = {
  low: {
    range: [0, 39],
    headline: "low visibility",
    body: "this clip is pretty kind to you. not many clear points to grab onto.",
  },
  medium: {
    range: [40, 69],
    headline: "moderate visibility",
    body: "you're gently glowing on radar. easy enough to follow, but not standing out.",
  },
  high: {
    range: [70, 100],
    headline: "high visibility",
    body: "you're basically a walking qr code here. everything about you is easy to lock onto.",
  },
} as const;

export type TrackingLevel = keyof typeof trackingSummary;

/**
 * Get the tracking level category based on score.
 */
export function getTrackingLevel(score: number): TrackingLevel {
  if (score >= 70) return "high";
  if (score >= 40) return "medium";
  return "low";
}

/**
 * Get the summary copy for a given score.
 */
export function getTrackingSummary(score: number) {
  const level = getTrackingLevel(score);
  return trackingSummary[level];
}

// =============================================================================
// TIPS SECTIONS
// "How to be less trackable" content for the bottom sheet
// =============================================================================

export const tipsSections = [
  {
    id: "phone",
    title: "phone camera & apps",
    icon: "📱",
    tips: [
      "if an app doesn't need your camera or mic, revoke that permission. it's usually in settings > privacy.",
      "cloud backup is comfy, but it also means every photo lives on someone else's computer forever.",
      "use approximate location instead of precise when you can—most apps work fine without knowing your exact spot.",
      "front-facing cameras are often lower res. that's not always a bad thing.",
    ],
  },
  {
    id: "public",
    title: "public cameras",
    icon: "🎥",
    tips: [
      "bright, even lighting makes you super easy to track. shadows, hats, and glasses add friction.",
      "being the only moving thing in a space puts you at the center of every algorithm's attention.",
      "patterns and reflective materials add visual noise that some trackers struggle with.",
      "look down or away from cameras when you can. direct face angles are gold for recognition.",
    ],
  },
  {
    id: "social",
    title: "social + mindset",
    icon: "🌐",
    tips: [
      "don't post the same angle and background forever—repetition makes linking you across platforms trivial.",
      "blur or crop friends who didn't ask to be online. you're protecting more than just yourself.",
      "metadata (time, location, device) often says more than the image itself. some apps strip it, some don't.",
      "you'll never be fully untrackable, but every bit of friction makes lazy systems struggle.",
    ],
  },
] as const;

export type TipsSection = typeof tipsSections[number];

// =============================================================================
// MISC COPY
// Small pieces used around the UI
// =============================================================================

export const miscCopy = {
  privacyNote: "no data stored. your video stays on your device.",
  processingHint: "counting how many ways your face can be recognized…",
  downloadNote: "video will be deleted from our servers after download.",
  learnMoreLink: "what does this mean?",
  tipsSheetTitle: "how to be less trackable",
  tipsSheetSubtitle: "a small survival guide for the camera age",
  
  // Alternate playback mode
  alternateModeLabel: "alternate with original",
  effectOnlyLabel: "effect only",
  alternateModeHelper: "see how much of you is still recognisable once reduced to data.",
} as const;

