"use client";

import { motion, AnimatePresence } from "framer-motion";

interface SequenceTimelineProps {
  sequence: string[];
  presets: { id: string; name: string; color?: string }[];
  onRemove: (index: number) => void;
  maxSlots: number;
}

export function SequenceTimeline({ sequence, presets, onRemove, maxSlots }: SequenceTimelineProps) {
  // Helper to get preset name
  const getPresetName = (id: string) => {
    const preset = presets.find((p) => p.id === id);
    return preset ? preset.name : id;
  };

  return (
    <div className="mb-4">
      <div className="flex justify-between items-center mb-2">
        <span className="text-text-muted text-xs font-mono uppercase tracking-widest">
          Sequence ({sequence.length}/{maxSlots})
        </span>
        {sequence.length === 0 && (
          <span className="text-text-muted text-xs italic opacity-50">
            tap effects to build chain
          </span>
        )}
      </div>

      <div className={`grid gap-2 h-12`} style={{ gridTemplateColumns: `repeat(${maxSlots}, 1fr)` }}>
        <AnimatePresence mode="popLayout">
          {sequence.map((effectId, index) => (
            <motion.div
              key={`${effectId}-${index}`}
              layout
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="relative group h-full"
            >
              <div className="h-full bg-white/10 border border-white/20 rounded-lg flex items-center justify-center px-1 overflow-hidden relative hover:bg-white/15 transition-colors duration-200">
                {/* Index indicator */}
                <span className="absolute top-1 left-1.5 text-[9px] text-text-muted font-mono opacity-50">
                  {index + 1}
                </span>

                <span className="text-[10px] md:text-xs text-white font-medium truncate px-1 text-center leading-tight">
                  {getPresetName(effectId)}
                </span>

                {/* Remove button overlay */}
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onRemove(index);
                  }}
                  className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity duration-200 active:bg-black/70"
                >
                  <span className="text-white text-xs font-bold scale-110">×</span>
                </button>
              </div>
              
              {/* Connecting line (visual only, if not last) */}
              {index < sequence.length - 1 && (
                <div className="absolute top-1/2 -right-2.5 w-3 h-[1px] bg-white/10 hidden md:block z-[-1]" />
              )}
            </motion.div>
          ))}
        </AnimatePresence>

        {/* Empty slots */}
        {Array.from({ length: Math.max(0, maxSlots - sequence.length) }).map((_, i) => (
          <div
            key={`empty-${i}`}
            className="h-full border border-white/5 rounded-lg flex items-center justify-center border-dashed"
          >
            <span className="text-white/10 text-xs">+</span>
          </div>
        ))}
      </div>
    </div>
  );
}

