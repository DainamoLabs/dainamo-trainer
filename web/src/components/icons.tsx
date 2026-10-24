import type { SVGProps } from "react";

const base = (d: React.ReactNode) => (p: SVGProps<SVGSVGElement>) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round" {...p}>
    {d}
  </svg>
);

export const IconPlus = base(<path d="M12 5v14M5 12h14" />);
export const IconQueue = base(<><path d="M4 6h16M4 12h16M4 18h10" /></>);
export const IconImages = base(<><rect x="3" y="3" width="18" height="18" rx="2" /><circle cx="9" cy="9" r="2" /><path d="m21 15-3.1-3.1a2 2 0 0 0-2.8 0L6 21" /></>);
export const IconLayers = base(<><path d="m12 2 9 5-9 5-9-5 9-5Z" /><path d="m3 12 9 5 9-5" /><path d="m3 17 9 5 9-5" /></>);
export const IconSettings = base(<><circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1Z" /></>);
export const IconFolder = base(<path d="M4 20h16a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.7-.9l-.8-1.2A2 2 0 0 0 7.9 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z" />);
export const IconFile = base(<><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2Z" /><path d="M14 2v6h6" /></>);
export const IconServer = base(<><rect x="3" y="3" width="18" height="5.5" rx="1.5" /><rect x="3" y="9.25" width="18" height="5.5" rx="1.5" /><rect x="3" y="15.5" width="18" height="5.5" rx="1.5" /><path d="M6.5 5.75h.01M9.25 5.75h.01M6.5 12h.01M9.25 12h.01M6.5 18.25h.01M9.25 18.25h.01" /></>);
export const IconDrive = base(<><path d="M2.5 13.5 5.3 6.3A2 2 0 0 1 7.2 5h9.6a2 2 0 0 1 1.9 1.3l2.8 7.2" /><rect x="2.5" y="13.5" width="19" height="6" rx="2" /><path d="M6.5 16.5h.01M9.5 16.5h.01" /></>);
export const IconDatabase = base(<><ellipse cx="12" cy="5" rx="8.5" ry="3" /><path d="M3.5 5v14c0 1.7 3.8 3 8.5 3s8.5-1.3 8.5-3V5" /><path d="M3.5 12c0 1.7 3.8 3 8.5 3s8.5-1.3 8.5-3" /></>);
export const IconPlayFill = base(<path d="M7.5 5.2v13.6a.8.8 0 0 0 1.2.7l10.9-6.8a.8.8 0 0 0 0-1.4L8.7 4.5a.8.8 0 0 0-1.2.7Z" fill="currentColor" />);
export const IconPause = base(<><rect x="6.5" y="5" width="3.5" height="14" rx="1" fill="currentColor" /><rect x="14" y="5" width="3.5" height="14" rx="1" fill="currentColor" /></>);
export const IconVolume = base(<><path d="M11 5 6.5 9H3.5v6h3l4.5 4V5Z" /><path d="M15.5 9a4.2 4.2 0 0 1 0 6M18.5 6a8.5 8.5 0 0 1 0 12" /></>);
export const IconMute = base(<><path d="M11 5 6.5 9H3.5v6h3l4.5 4V5Z" /><path d="m21.5 9.5-5 5M16.5 9.5l5 5" /></>);
export const IconUp = base(<path d="m18 15-6-6-6 6" />);
export const IconDown = base(<path d="m6 9 6 6 6-6" />);
export const IconLeft = base(<path d="m15 18-6-6 6-6" />);
export const IconRight = base(<path d="m9 18 6-6-6-6" />);
export const IconX = base(<path d="M18 6 6 18M6 6l12 12" />);
export const IconTrash = base(<><path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6" /></>);
export const IconStop = base(<rect x="6" y="6" width="12" height="12" rx="1.5" />);
export const IconPlay = base(<path d="m7 4 13 8-13 8V4Z" />);
export const IconRefresh = base(<><path d="M3 12a9 9 0 0 1 15.5-6.3L21 8" /><path d="M21 3v5h-5" /><path d="M21 12a9 9 0 0 1-15.5 6.3L3 16" /><path d="M3 21v-5h5" /></>);
export const IconSparkle = base(<path d="M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9L12 3ZM19 16l.8 2.2L22 19l-2.2.8L19 22l-.8-2.2L16 19l2.2-.8L19 16Z" />);
export const IconDownload = base(<><path d="M12 3v12M7 10l5 5 5-5" /><path d="M5 21h14" /></>);
export const IconCamera = base(<><path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3Z" /><circle cx="12" cy="13" r="3" /></>);
export const IconSave = base(<><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2Z" /><path d="M17 21v-8H7v8M7 3v5h8" /></>);
export const IconUpload = base(<><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><path d="m17 8-5-5-5 5M12 3v12" /></>);
export const IconCopy = base(<><rect x="9" y="9" width="13" height="13" rx="2" /><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" /></>);
export const IconCheck = base(<path d="M20 6 9 17l-5-5" />);
export const IconAlert = base(<><circle cx="12" cy="12" r="10" /><path d="M12 8v4M12 16h.01" /></>);
export const IconTarget = base(<><circle cx="12" cy="12" r="9" /><circle cx="12" cy="12" r="5" /><circle cx="12" cy="12" r="1" /></>);
export const IconScissors = base(<><circle cx="6" cy="6" r="3" /><circle cx="6" cy="18" r="3" /><path d="M20 4 8.1 15.9M14.5 14.5 20 20M8.1 8.1 12 12" /></>);
export const IconEdit = base(<><path d="M12 20h9" /><path d="M16.5 3.5a2.1 2.1 0 1 1 3 3L7 19l-4 1 1-4Z" /></>);
export const IconMore = base(<path d="M5 12h.01M12 12h.01M19 12h.01" strokeWidth={3.6} />);
export const IconSearch = base(<><circle cx="11" cy="11" r="7" /><path d="m20 20-3.5-3.5" /></>);
export const IconArrowRight = base(<path d="M5 12h14M13 6l6 6-6 6" />);
export const IconLink = base(<><path d="M10 13a5 5 0 0 0 7.5.5l3-3a5 5 0 0 0-7-7l-1.7 1.7" /><path d="M14 11a5 5 0 0 0-7.5-.5l-3 3a5 5 0 0 0 7 7l1.7-1.7" /></>);
export const IconGrid = base(<><rect x="3.5" y="3.5" width="7" height="7" rx="1.5" /><rect x="13.5" y="3.5" width="7" height="7" rx="1.5" /><rect x="3.5" y="13.5" width="7" height="7" rx="1.5" /><rect x="13.5" y="13.5" width="7" height="7" rx="1.5" /></>);
export const IconList = base(<><path d="M9 6h11M9 12h11M9 18h11" /><path d="M4.5 6h.01M4.5 12h.01M4.5 18h.01" strokeWidth={2.6} /></>);
export const IconCheckSquare = base(<><rect x="3.5" y="3.5" width="17" height="17" rx="3" /><path d="m8.5 12 2.5 2.5 4.5-5" /></>);
export const IconCheckCircle = base(<><circle cx="12" cy="12" r="10" fill="currentColor" stroke="none" /><path d="m7.5 12.2 3 3 6-6.2" stroke="#0b0b0b" strokeWidth={2.4} /></>);
export const IconWave = base(<path d="M4 10v4M8 7v10M12 4v16M16 8v8M20 10v4" />);
export const IconFilm = base(<><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M7 4v16M17 4v16M3 9h4M3 15h4M17 9h4M17 15h4" /></>);
export const IconInfo = base(<><circle cx="12" cy="12" r="10" /><path d="M12 16v-4.5M12 8h.01" /></>);
export const IconMinusCircle = base(<><circle cx="12" cy="12" r="9" /><path d="M8 12h8" /></>);
export const IconClock = base(<><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></>);
export const IconQueueAdd = base(<><path d="M3 6h14M3 12h10M3 18h7" /><path d="M14 18h7M18 15l3 3-3 3" /></>);
export const IconUpscale = base(<><path d="M15 3h6v6" /><path d="M9 21H3v-6" /><path d="m21 3-7 7" /><path d="m3 21 7-7" /></>);
export const IconTools = base(<path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z" />);
