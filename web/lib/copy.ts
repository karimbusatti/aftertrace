export interface TipItem {
  text: string;
  impact: "high" | "medium" | "low";
}

export interface TipSection {
  id: string;
  title: string;
  icon: string;
  intro: string;
  tips: TipItem[];
}

export const tipsSections: TipSection[] = [
  {
    id: "devices",
    title: "Lock down your devices",
    icon: "📱",
    intro: "The camera you carry is the one watching you most. Start here.",
    tips: [
      { text: "Audit camera & mic permissions: Settings → Privacy. Revoke access from every app that doesn't strictly need it.", impact: "high" },
      { text: "Set location to \"While Using\", never \"Always\". Most apps don't need your background location at all.", impact: "high" },
      { text: "Turn off photo cloud sync, or use end-to-end encrypted storage. Faceprints get built from cloud libraries.", impact: "medium" },
      { text: "Strip EXIF metadata before sharing, photos carry GPS coordinates, device IDs and timestamps.", impact: "medium" },
      { text: "Physically cover the front camera when idle. Low-tech, unbeatable.", impact: "low" },
    ],
  },
  {
    id: "public",
    title: "Move through public space",
    icon: "🚶",
    intro: "Cameras read your face, your walk, and your routine. Break the pattern.",
    tips: [
      { text: "Break up facial geometry: brimmed hats, bulky frames, and asymmetric hair defeat many detectors.", impact: "high" },
      { text: "Vary your routes and timing. Pattern-of-life tracking links you across cameras by predictability.", impact: "high" },
      { text: "Don't look up at cameras. A clean frontal frame is exactly what recognition wants.", impact: "medium" },
      { text: "Reflective or IR-blocking eyewear can wash out your face on IR-sensitive cameras.", impact: "medium" },
      { text: "Pay cash where it matters. Payment data ties a face to an identity instantly.", impact: "low" },
    ],
  },
  {
    id: "digital",
    title: "Shrink your digital trail",
    icon: "🌐",
    intro: "Every public photo of your face is training data for someone's model.",
    tips: [
      { text: "Lock down or remove public face photos. Reverse image search links profiles in seconds.", impact: "high" },
      { text: "Use a different photo, handle, and email per service. Linkability is what makes you trackable.", impact: "high" },
      { text: "Kill geotags, check-ins, and location tags. They reconstruct your movements after the fact.", impact: "medium" },
      { text: "Opt out of data brokers (Clearview, PimEyes, etc. honor removal requests in many regions).", impact: "medium" },
      { text: "Assume anything posted is permanent and scrapable, even after you delete it.", impact: "low" },
    ],
  },
  {
    id: "awareness",
    title: "Know the threat model",
    icon: "🧠",
    intro: "You can't opt out of everything, so spend effort where it counts.",
    tips: [
      { text: "Most surveillance is automated logging, not a person watching. The risk is the permanent record.", impact: "high" },
      { text: "Your faceprint can't be changed like a password. Protect it like one you can never reset.", impact: "high" },
      { text: "Gait, posture, tattoos, and clothing are all secondary identifiers when your face is hidden.", impact: "medium" },
      { text: "Voice is biometric too, assume always-on mics can fingerprint how you speak.", impact: "medium" },
      { text: "The goal isn't invisibility. It's raising the cost of tracking until you're not worth it.", impact: "low" },
    ],
  },
];

// Rotating facts shown while a clip is processing. (No em dashes.)
export const surveillanceFacts: string[] = [
  "Modern face recognition can identify you from a single frame, even partially turned or in a crowd.",
  "The way you walk is nearly as unique as a fingerprint, and trackable from low resolution CCTV at a distance.",
  "Most public faceprint databases were built by scraping photos people posted publicly, including yours.",
  "Live facial recognition can scan thousands of faces a minute and flag matches in real time.",
  "Features around your eyes stay identifiable even when you are wearing a mask.",
  "Once your faceprint is in a database, you cannot change it like a password. It is permanent.",
  "Some retailers track how long you look at a product and link your visits across different stores.",
  "Border and airport systems increasingly match your live face to your passport photo automatically.",
  "Emotion detection AI claims to read your mood from micro expressions, and it is frequently wrong.",
  "Citywide camera networks can hand a single person off from one camera to the next, automatically.",
  "Even blurred or pixelated faces can sometimes be reconstructed by AI upscalers.",
  "Your phone can be fingerprinted and tied to your face through tagged photos you never uploaded.",
  "A face only needs to be a few dozen pixels wide for some systems to attempt a match.",
  "Recognition models keep working in the dark using infrared and near infrared camera feeds.",
  "Your unique pattern of veins, freckles, and skin texture can be enough to tell you apart.",
  "Many doorbell and dashcam clips are uploaded to clouds that allow third party face search.",
  "Social platforms can detect faces in the background of other people's photos and tag you.",
  "3D face maps from phone unlock sensors describe the exact geometry of your face.",
  "Gait, body shape, and clothing let systems re identify you after your face leaves the frame.",
  "Training a model to recognize you can take as few as a handful of clear images.",
  "Optical flow lets a camera track exactly how every part of you moves between frames.",
  "A point cloud turns your body into thousands of 3D coordinates that can be stored forever.",
  "Crowd cameras can count, track, and follow individuals through a packed venue in real time.",
  "Number plate readers log where your car has been, building a timeline of your movements.",
  "Wifi and bluetooth signals from your phone can track your path through a building without a camera.",
  "Smart billboards can detect your age, gender, and attention, then change the ad just for you.",
  "Thermal cameras can pick out a person in total darkness from their body heat alone.",
  "Depth sensors see the shape of your face even behind a flat photo, so pictures cannot fool them.",
  "Your typing rhythm and how you hold your phone can identify you without any camera at all.",
  "Deleting a photo rarely deletes the copies that backups, caches, and scrapers already made.",
  "The more unique your look, the easier you are to single out in a crowd of faces.",
];

export function getPersonalizedTips(score: number): string[] {
  if (score >= 70) {
    return [
      "Your video has high trackability. Multiple clear frames of identifying features.",
      "Consider recording in lower light or with partial face occlusion.",
      "Reduce motion stability, smooth, predictable movement is easier to track.",
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
