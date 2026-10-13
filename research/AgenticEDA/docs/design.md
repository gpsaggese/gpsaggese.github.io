# Design System: Neural Slate

 

### 1. Overview & Creative North Star

**Creative North Star: "The Intelligent Canvas"**

 

Neural Slate is a high-density, analytical design system built for complex data orchestration and agentic workflows. It rejects the "app-as-a-toy" aesthetic in favor of a "Pro-Tool" environment that feels like a precision instrument. The system is characterized by **Structured Density**, using a monochromatic base punctuated by high-fidelity primary blues and intentional pops of warmth. It leverages a rigorous technical layout that balances the fluidity of AI interactions with the rigid structure of data analysis. The spacing in Neural Slate is now more generous, set to `2` to provide a normal, balanced amount of whitespace.

 

### 2. Colors

The color palette is rooted in a "Cold-to-Electric" spectrum.

- **Primary Roles:** The primary blue (`#004ac6`) serves as the "Action Thread," used for active states, running processes, and critical CTAs.

- **Secondary/Tertiary:** Subdued indigos and warm oranges are reserved for stateful metadata (e.g., Churn indicators or AI suggestions).

- **The "No-Line" Rule:** Visual separation is achieved through background shifts (e.g., `surface-container-low` for sidebars vs. `surface-container-lowest` for the main workspace). Traditional borders are replaced by 1px shifts in tonal value or 10% opacity variants of the outline color.

- **Surface Hierarchy:** 

    - **Lowest:** The work area (canvas).

    - **Low/Medium:** Navigation rails and tool panels.

    - **High/Highest:** Hover states and active selections.

- **Signature Textures:** Use `primary-to-primary-container` gradients for Floating Action Buttons (FABs) to provide a tactile, high-contrast focal point against the matte workspace.

 

### 3. Typography

Neural Slate utilizes a dual-font strategy to distinguish between "Command" and "Content."

- **Manrope (Headline):** Used for structural identity and high-level headers. It provides a technical yet modern geometric feel.

- **Inter (Body/Label):** The workhorse for data density. Used for all UI controls and labels.

- **JetBrains Mono (Code):** Essential for the "Notebook" context, used for input/output and system logs.

 

**Typographic Scale (Ground Truth):**

- **Display/Hero:** 2.25rem (36px) - Heavy weighting for main titles.

- **Section Headers:** 1.125rem (18px) - Bold tracking.

- **Body Text:** 0.875rem (14px) - Standardized for readability.

- **Technical UI:** 0.75rem (12px) down to 10px - Used for explorer items, labels, and timestamps to maintain extreme density without sacrificing clarity.

 

### 4. Elevation & Depth

Depth is communicated through **Tonal Stacking** rather than heavy shadows.

- **The Layering Principle:** Panels are "pressed" into the background using `surface-container-low`. Active content "pops" using `surface-container-lowest`.

- **Ambient Shadows:** The `code-shadow` (0px 4px 12px rgba(11, 28, 48, 0.04)) is the standard for floating cards and code blocks, providing a "lift" that feels like paper on a desk rather than a heavy object.

- **Glassmorphism:** Navigation headers and top bars should use a subtle backdrop blur with a semi-transparent white or slate background to maintain context of the content scrolling beneath.

 

### 5. Components

- **Buttons:** 

    - *Primary:* Solid fills with uppercase, tracked-out text (10px).

    - *Ghost:* 1px `outline-variant/30` with bold, uppercase labels.

- **The Timeline/Agent Action:** A bespoke component using a vertical track and pulse-indicators to show real-time AI "thinking" states.

- **Code Cells:** Nested containers with a distinct line-number gutter (`surface-container-highest`) and a low-elevation shadow.

- **Bento-style Action Cards:** Used in the chat sidebar for quick AI commands, utilizing `surface-container-lowest` and hover-active primary tints.

- **Status Indicators:** Micro-pills with 0.5rem (8px) typography for "In [1]" or status tags.

 

### 6. Do's and Don'ts

**Do:**

- Use uppercase and letter-spacing for "Meta" labels (Explorer, Agent Actions).

- Maintain high density; keep padding tight to allow for maximum data visibility.

- Use `animate-pulse` for active system processes.

- Align code and data to a strict baseline.

 

**Don't:**

- Do not use rounded corners larger than 0.5rem (8px) for structural containers; maintain a professional, slightly "sharp" edge.

- Do not use high-saturation backgrounds for large areas; keep the workspace "Slate" (Neutral-Blue) to reduce eye strain.

- Avoid 1px solid black borders; always use the `outline-variant` at low opacity.