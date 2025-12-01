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
    explainer: "how easy it is to follow you across frames. higher = more visible to algorithms.",
  },
  longest_track_seconds: {
    label: "longest track",
    explainer: "how long a single point on you stayed locked. longer = steadier signal.",
  },
  max_continuous_tracking_frames: {
    label: "max frames tracked",
    explainer: "the most frames a single point survived. more = more data to work with.",
  },
  average_points_per_frame: {
    label: "avg features",
    explainer: "trackable points found on you at any moment. more = richer profile.",
  },
  people_detected: {
    label: "subjects",
    explainer: "distinct people detected in frame. each one is a separate data trail.",
  },
  frames_processed: {
    label: "frames",
    explainer: "total video frames analyzed. each frame is a snapshot of your biometric surface.",
  },
  total_points_spawned: {
    label: "total points",
    explainer: "all trackable features spawned and followed throughout the clip.",
  },
  beats_detected: {
    label: "audio beats",
    explainer: "audio peaks used to sync visual effects. no beats means we guessed timing.",
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
    body: "you're blending in. not many clear points to grab onto.",
  },
  medium: {
    range: [40, 69],
    headline: "moderate visibility",
    body: "you're on radar. followable, but not standing out.",
  },
  high: {
    range: [70, 100],
    headline: "high visibility",
    body: "you're fully legible. everything about you is easy to lock onto.",
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
// PERSONALIZED TIPS
// Tips tailored to the user's trackability score
// =============================================================================

export const personalizedTips = {
  low: [
    "you're already hard to track. keep doing what you're doing.",
    "low lighting and motion blur are working in your favor here.",
    "consider this your baseline — you can use this as a reference for other clips.",
  ],
  medium: [
    "your face is visible for extended periods — consider angles that break the frontal view.",
    "consistent lighting makes tracking easier. shadows and movement help.",
    "try adding visual noise: hats, glasses, hair across your face.",
    "if you're in frame alone, you're the only thing to focus on.",
  ],
  high: [
    "you're very trackable here. clear face, stable position, good lighting — all things algorithms love.",
    "frontal face exposure is the main factor. turn slightly, tilt your head, add occlusion.",
    "if this were actual surveillance footage, you'd be trivial to identify.",
    "consider what you're wearing: plain, solid colors make you easier to isolate.",
    "you're in frame alone with clear contrast. crowded scenes break the signal.",
  ],
} as const;

export function getPersonalizedTips(score: number): string[] {
  const level = getTrackingLevel(score);
  return [...personalizedTips[level]];
}

// =============================================================================
// TIPS SECTIONS
// "How to be less trackable" content for the bottom sheet
// =============================================================================

export const tipsSections = [
  {
    id: "devices",
    title: "your devices",
    tips: [
      "revoke camera/mic permissions from apps that don't need them. settings → privacy.",
      "cloud backup = your photos on corporate servers indefinitely. local or encrypted only.",
      "approximate location works for 90% of apps. precise GPS is a gift you don't owe.",
      "front cameras often have lower resolution. sometimes that's a feature, not a bug.",
      "clear your photo metadata before uploading. EXIF data includes location, device, time.",
    ],
  },
  {
    id: "physical",
    title: "physical space",
    tips: [
      "flat, even lighting is optimal for facial recognition. shadows break the pattern.",
      "glasses, hats, hair across your face — anything that fragments the face oval helps.",
      "move with crowds, not through empty spaces. being the only motion source is a spotlight.",
      "patterned clothing and asymmetric accessories create visual noise for body tracking.",
      "infrared LEDs are invisible to you but can blind cameras at night. legal gray area.",
    ],
  },
  {
    id: "digital",
    title: "digital hygiene",
    tips: [
      "same angle, same background, same pose = trivial to link across platforms.",
      "metadata (timestamp, GPS, device ID) often reveals more than the image itself.",
      "reverse image search yourself occasionally. see what's already out there.",
      "friends in your photos inherit your privacy choices. blur faces without consent.",
      "100% untrackable isn't the goal. friction is. make it expensive to follow you.",
    ],
  },
  {
    id: "awareness",
    title: "general awareness",
    tips: [
      "public cameras are everywhere. entrances, intersections, transit, stores. map them mentally.",
      "smart devices are always listening: phones, speakers, TVs. assume the mic is on.",
      "your gait, posture, and movement patterns are as unique as your fingerprint.",
      "facial recognition works from surprisingly far away. 50+ meters with good cameras.",
      "the goal isn't paranoia. it's informed consent about your own visibility.",
    ],
  },
] as const;

export type TipsSection = (typeof tipsSections)[number];

// =============================================================================
// MISC COPY
// Small pieces used around the UI
// =============================================================================

export const miscCopy = {
  privacyNote: "no data stored. your video stays on your device.",
  processingHint: "mapping your biometric surface...",
  downloadNote: "video will be deleted from our servers after download.",
  learnMoreLink: "what does this mean?",
  tipsSheetTitle: "counter-surveillance",
  tipsSheetSubtitle: "practical friction for the camera age",
  
  // Alternate playback mode
  alternateModeLabel: "compare",
  effectOnlyLabel: "effect",
  alternateModeHelper: "see how much of you is still recognizable once reduced to data.",
} as const;
