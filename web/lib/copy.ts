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
    id: "devices",
    title: "your devices",
    icon: null,
    tips: [
      "revoke camera/mic permissions from apps that don't need them. settings > privacy.",
      "cloud backup means your photos live on corporate servers indefinitely.",
      "approximate location works for 90% of apps. precise location is a gift you don't owe anyone.",
      "front cameras have lower resolution. sometimes that's a feature, not a bug.",
      "clear your photo metadata before uploading. exiftool or shortcut automations work.",
    ],
  },
  {
    id: "physical",
    title: "physical space",
    icon: null,
    tips: [
      "flat, even lighting is optimal for facial recognition. shadows break the pattern.",
      "glasses, hats, hair across your face—anything that fragments the oval helps.",
      "move with crowds, not through empty spaces. being the only motion source is a spotlight.",
      "patterned clothing and asymmetric accessories create visual noise.",
      "infrared LEDs are invisible to you but can blind cameras at night.",
    ],
  },
  {
    id: "digital",
    title: "digital hygiene",
    icon: null,
    tips: [
      "same angle, same background, same pose = trivial to link across platforms.",
      "metadata (timestamp, GPS, device ID) often reveals more than the image itself.",
      "reverse image search yourself occasionally. see what's already out there.",
      "friends in your photos inherit your privacy choices. blur faces without consent.",
      "100% untrackable isn't the goal. friction is. make it expensive to follow you.",
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
  processingHint: "mapping your biometric surface…",
  downloadNote: "video will be deleted from our servers after download.",
  learnMoreLink: "what does this mean?",
  tipsSheetTitle: "counter-surveillance basics",
  tipsSheetSubtitle: "practical friction for the camera age",
  
  // Alternate playback mode
  alternateModeLabel: "alternate with original",
  effectOnlyLabel: "effect only",
  alternateModeHelper: "see how much of you is still recognisable once reduced to data.",
} as const;

