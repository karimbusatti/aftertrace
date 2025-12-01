export const tipsSections = [
  {
    id: "devices",
    title: "Device Hygiene",
    tips: [
      "Revoke camera and microphone permissions from apps that don't need them. Settings → Privacy → Camera/Microphone.",
      "Disable location services entirely, or set to 'While Using' for essential apps only.",
      "Turn off iCloud Photo sync. Your photos shouldn't live on servers you don't control.",
      "Cover your front camera when not in use. Low-tech, high-impact.",
      "Regularly clear your photo metadata with exiftool or automation shortcuts before sharing.",
    ],
  },
  {
    id: "public",
    title: "In Public Spaces",
    tips: [
      "Avoid looking directly at cameras. Gaze direction is a tracking vector.",
      "Wear hats with brims, scarves, or glasses to obscure facial geometry.",
      "IR LEDs embedded in glasses can blind cameras without being visible to humans.",
      "Walk unpredictably. Gait analysis can identify you from body movement alone.",
      "Avoid touching surfaces unnecessarily. Fingerprints persist longer than you think.",
    ],
  },
  {
    id: "digital",
    title: "Digital Presence",
    tips: [
      "Blur or crop your face from photos before posting anywhere.",
      "Never use the same profile photo across platforms. Reverse image search is trivial.",
      "Avoid check-ins, location tags, and geotagged posts entirely.",
      "Use different usernames and emails per service. Linkability is the enemy.",
      "Assume any camera-connected device is recording, always.",
    ],
  },
  {
    id: "awareness",
    title: "Threat Awareness",
    tips: [
      "Most surveillance is ambient and automated. You're not being watched, you're being logged.",
      "Facial recognition databases are built from public photos. Every upload is a contribution.",
      "License plates, tattoos, and clothing patterns are all tracking vectors.",
      "Audio fingerprinting can identify you from voice alone. Consider when you speak near devices.",
      "The goal isn't invisibility. It's raising the cost of surveillance to make you not worth tracking.",
    ],
  },
];

export function getPersonalizedTips(score: number): string[] {
  if (score >= 70) {
    return [
      "Your video has high trackability. Multiple clear frames of identifying features.",
      "Consider recording in lower light or with partial face occlusion.",
      "Reduce motion stability — smooth, predictable movement is easier to track.",
      "Avoid static backgrounds that make subject isolation trivial.",
    ];
  } else if (score >= 40) {
    return [
      "Moderate trackability. Some frames contain clear identifying features.",
      "Quick movements and partial occlusions are working in your favor.",
      "Consider more dynamic backgrounds to complicate subject extraction.",
      "Environmental noise (other people, movement) helps obscure you.",
    ];
  } else {
    return [
      "Low trackability. This video would be challenging to process automatically.",
      "Motion blur, occlusions, or poor lighting are reducing detection confidence.",
      "This is the baseline to aim for when privacy matters.",
      "Remember: even low trackability isn't zero. Persistent surveillance accumulates.",
    ];
  }
}

export function getTrackingLevel(score: number): { level: string; message: string } {
  if (score >= 70) {
    return {
      level: "high",
      message: "High trackability. You'd be easy to identify and follow in this footage.",
    };
  } else if (score >= 40) {
    return {
      level: "medium", 
      message: "Moderate trackability. Some identifying features are visible, but not consistently.",
    };
  } else {
    return {
      level: "low",
      message: "Low trackability. Automated systems would struggle with this footage.",
    };
  }
}

export const miscCopy = {
  processingHint: "Frame-by-frame analysis takes about 30 seconds",
  tipsSheetTitle: "Reduce your footprint",
  privacyNote: "No data stored · Deleted after download",
};
